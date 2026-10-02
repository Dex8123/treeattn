import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn.attention_dyn import DynAttention
from treeattn.config import Config
from treeattn.engine import Decoder, weights_from_ours
from treeattn.model import GPT
from treeattn.seg import level_sums, select_seg


def tiny(**kw):
    base = dict(vocab_size=64, seq_len=64, n_layer=2, n_head=2, n_embd=32, attn='seg', wmax=4, beam=2,
                beam_classes=(2, 3, 4), nb_window=4, nb_store=2, nb_probe=2, nb_cap_min=1)
    base.update(kw)
    return Config(**base)


def test_levels_additive():
    torch.manual_seed(0)
    k = torch.randn(1, 2, 16, 4)
    S = level_sums(k, 3, False, 0.25)
    for l in (1, 2, 3):
        want = k.reshape(1, 2, 16 >> l, 1 << l, 4).sum(3)
        assert (S[l] - want).abs().max() < 1e-5, l


def test_past_is_covered_exactly():
    """With a wide beam, tree keys + static keys must cover every position 0..t (and nothing later)."""
    torch.manual_seed(1)
    cfg = tiny(wmax=32, beam=32, neighbors=False, beam_mode='fixed')
    T = 64
    q, k = torch.randn(1, 1, T, 16), torch.randn(1, 1, T, 16)
    sel = select_seg(q, k, torch.full((1, 1, T), 32, dtype=torch.long), cfg)
    for t in range(T):
        pos = set(sel['tree_pos'][0, 0, t][sel['tree_pos'][0, 0, t] >= 0].tolist())
        assert all(p < t for p in pos), ('future key selected', t)
        got = pos | {0, t, t - 1, t - 2}
        got = {p for p in got if p >= 0}
        assert got == set(range(t + 1)), (t, sorted(set(range(t + 1)) - got))


def test_all_keys_matches_dense():
    torch.manual_seed(2)
    cfg = tiny(n_layer=1, wmax=32, beam=32, neighbors=False, beam_mode='fixed')
    attn = DynAttention(cfg).eval()
    x = torch.randn(2, 64, 32)
    with torch.no_grad():
        out = attn(x)
        C, H, dh = 32, 2, 16
        q, k, v = attn.c_attn(x).split(C, 2)
        q, k, v = [t.view(2, 64, H, dh).transpose(1, 2) for t in (q, k, v)]
        s = (q @ k.transpose(-1, -2)) / dh ** 0.5
        s = s.masked_fill(~torch.tril(torch.ones(64, 64, dtype=torch.bool)), float('-inf'))
        ref = attn.c_proj((torch.softmax(s, -1) @ v).transpose(1, 2).reshape(2, 64, C))
    err = (out - ref).abs().max().item()
    print('  max difference to dense attention: %.2e' % err)
    assert err < 1e-4, err


def test_causality():
    for kw in [dict(beam_mode='fixed'), dict(beam_mode='pred', global_pool=True), dict(beam_mode='pred', gate=True, local=8)]:
        torch.manual_seed(3)
        model = GPT(tiny(**kw)).eval()
        idx = torch.randint(0, 64, (2, 64))
        idx2 = idx.clone()
        idx2[:, 37] = (idx[:, 37] + 1) % 64
        with torch.no_grad():
            l1, _, _ = model(idx)
            l2, _, _ = model(idx2)
        before = (l1[:, :37] - l2[:, :37]).abs().max().item()
        after = (l1[:, 37:] - l2[:, 37:]).abs().max().item()
        print('  %s: change before pos %.2e, after pos %.2e' % (kw, before, after))
        assert before < 1e-4, ('LEAK', kw, before)
        assert after > 1e-6, ('token change had no effect', kw)


def test_engine_matches_forward():
    """Token-by-token decoding with the key cache must give the same logits as the training-mode forward."""
    for kw in [dict(beam_mode='fixed'), dict(beam_mode='pred', global_pool=True, gate=True)]:
        torch.manual_seed(4)
        cfg = tiny(**kw)
        model = GPT(cfg).eval()
        idx = torch.randint(0, 64, (1, 64))
        with torch.no_grad():
            ref = model(idx)[0][0]
            dec = Decoder(weights_from_ours(model.state_dict(), cfg), cfg.n_layer, cfg.n_head, cfg.n_embd, 64, 'seg', cfg, 'cpu')
            got = torch.stack([dec.step(int(idx[0, t]), t)[0] for t in range(64)])
        err = (ref - got).abs().max().item()
        print('  %s: max logit difference engine vs forward %.2e' % (kw, err))
        assert err < 1e-3, err


def test_backward_and_losses():
    torch.manual_seed(5)
    model = GPT(tiny(beam_mode='pred', global_pool=True, gate=True)).train()
    idx = torch.randint(0, 64, (2, 64))
    _, loss, st = model(idx, torch.randint(0, 64, (2, 64)))
    loss.backward()
    print('  lm %.3f pred %.3f | %s' % (float(st['lm']), st['pred'], {k_: round(v_, 2) for k_, v_ in st.items()
                                                                       if k_ not in ('lm', 'pred', 'route')}))
    assert torch.isfinite(loss) and st['pred'] > 0
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


def test_fullsize_causality():
    if not torch.cuda.is_available():
        print('  skipped (no GPU)')
        return
    cfg = Config(attn='seg', beam_mode='pred', global_pool=True, gate=True)
    torch.manual_seed(6)
    model = GPT(cfg).cuda().eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len), device='cuda')
    with torch.no_grad():
        l1, _, _ = model(idx)
        for pos in (70, 200, 333, 500):
            idx2 = idx.clone()
            idx2[:, pos] = (idx[:, pos] + 1) % cfg.vocab_size
            l2, _, _ = model(idx2)
            before = (l1[:, :pos] - l2[:, :pos]).abs().max().item()
            print('  pos %d: change before %.2e' % (pos, before))
            assert before < 1e-3, ('LEAK', pos, before)


if __name__ == '__main__':
    failed = 0
    for fn in [test_levels_additive, test_past_is_covered_exactly, test_all_keys_matches_dense, test_causality,
               test_engine_matches_forward, test_backward_and_losses, test_fullsize_causality]:
        try:
            fn()
            print('PASS', fn.__name__)
        except Exception:
            failed += 1
            print('FAIL', fn.__name__)
            traceback.print_exc()
    print('ALL TESTS PASSED' if failed == 0 else '%d TEST(S) FAILED' % failed)
    sys.exit(1 if failed else 0)
