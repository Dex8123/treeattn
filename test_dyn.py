import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn.attention_dyn import DynAttention
from treeattn.config import Config
from treeattn.dyntree import LaneTree, select_keys
from treeattn.model import GPT


def tiny(**kw):
    base = dict(vocab_size=64, seq_len=64, n_layer=2, n_head=2, n_embd=32, attn='dyn', chunk=16, wmax=4, beam=2,
                beam_classes=(2, 3, 4), nb_window=4, nb_store=2, nb_probe=2, nb_cap_min=1)
    base.update(kw)
    return Config(**base)


def test_tree_insertion_consistent():
    """After many insertions every node sum/count must equal a recomputation from the leaf slots."""
    torch.manual_seed(0)
    P, C, dh = 4, 64, 8
    kfull = torch.randn(P, C, dh)
    n = torch.tensor([0, 8, 20, 33])
    tree = LaneTree(kfull, n, iters=3)
    assert tree.check() < 1e-4, 'initial build inconsistent'
    ar = torch.arange(P)
    for i in range(16):
        kpos = n + i
        tree.insert(kfull[ar, kpos], kpos, kpos >= 1)
    worst = tree.check()
    print('  worst sum/count mismatch after insertions: %.2e' % worst)
    assert worst < 1e-4, worst
    pos = tree.slot.reshape(P, C)
    for p in range(P):
        got = sorted(pos[p][pos[p] >= 0].tolist())
        want = list(range(1, int(n[p]) + 16))
        assert got == want, (p, got[:5], want[:5])


def test_selection_is_causal():
    """Every selected key must lie strictly before the query."""
    torch.manual_seed(1)
    cfg = tiny()
    b, H, T, dh = 2, 2, 64, 16
    q, k = torch.randn(b, H, T, dh), torch.randn(b, H, T, dh)
    lhat = torch.full((b, H, T), 3, dtype=torch.long)
    sel = select_keys(q, k, lhat, cfg)
    t = torch.arange(T).view(1, 1, T, 1)
    for name in ('tree_pos', 'nb_pos'):
        pos = sel[name]
        assert ((pos < t) | (pos < 0)).all(), name
    print('  tree keys per query %.1f, neighbour keys per query %.1f' % (
        (sel['tree_pos'] >= 0).sum(-1).float().mean(), (sel['nb_pos'] >= 0).sum(-1).float().mean()))


def test_all_keys_matches_dense():
    """With every key selected the sparse attention must equal ordinary causal attention."""
    torch.manual_seed(2)
    cfg = tiny(n_layer=1, wmax=32, beam=32, neighbors=False, local=3, beam_mode='fixed')
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
    """Changing a token must not change the output at any EARLIER position."""
    for kw in [dict(beam_mode='fixed'), dict(beam_mode='pred', global_pool=True), dict(beam_mode='pred', local=8)]:
        torch.manual_seed(3)
        model = GPT(tiny(**kw)).eval()
        idx = torch.randint(0, 64, (2, 64))
        idx2 = idx.clone()
        pos = 37
        idx2[:, pos] = (idx[:, pos] + 1) % 64
        with torch.no_grad():
            l1, _, _ = model(idx)
            l2, _, _ = model(idx2)
        before = (l1[:, :pos] - l2[:, :pos]).abs().max().item()
        after = (l1[:, pos:] - l2[:, pos:]).abs().max().item()
        print('  %s: change before pos %.2e, after pos %.2e' % (kw, before, after))
        assert before < 1e-4, ('LEAK', kw, before)
        assert after > 1e-6, ('token change had no effect', kw)


def test_backward_and_losses():
    torch.manual_seed(4)
    model = GPT(tiny(beam_mode='pred', global_pool=True)).train()
    idx = torch.randint(0, 64, (2, 64))
    tgt = torch.randint(0, 64, (2, 64))
    _, loss, st = model(idx, tgt)
    loss.backward()
    print('  lm %.3f pred %.3f | %s' % (float(st['lm']), st['pred'], {k_: round(v_, 2) for k_, v_ in st.items()
                                                                       if k_ not in ('lm', 'pred', 'route')}))
    assert torch.isfinite(loss)
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
    assert model.blocks[0].attn.c_attn.weight.grad.abs().sum() > 0
    assert st['pred'] > 0


def test_fullsize_causality():
    """Same leak check at the real model size (needs a GPU)."""
    if not torch.cuda.is_available():
        print('  skipped (no GPU)')
        return
    cfg = Config(attn='dyn', beam_mode='pred', global_pool=True)
    torch.manual_seed(5)
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
    for fn in [test_tree_insertion_consistent, test_selection_is_causal, test_all_keys_matches_dense,
               test_causality, test_backward_and_losses, test_fullsize_causality]:
        try:
            fn()
            print('PASS', fn.__name__)
        except Exception:
            failed += 1
            print('FAIL', fn.__name__)
            traceback.print_exc()
    print('ALL TESTS PASSED' if failed == 0 else '%d TEST(S) FAILED' % failed)
    sys.exit(1 if failed else 0)
