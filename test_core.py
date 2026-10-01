import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn.attention import TreeAttention
from treeattn.config import Config
from treeattn.model import GPT


def tiny(**kw):
    base = dict(vocab_size=64, seq_len=128, n_layer=2, n_head=2, n_embd=32, tree_block=32, wmax=4, beam=2,
                beam_classes=(2, 3, 4))
    base.update(kw)
    return Config(**base)


def test_all_keys_matches_dense():
    """With every key selected the sparse attention must equal ordinary causal attention."""
    torch.manual_seed(0)
    cfg = tiny(n_layer=1, wmax=64, beam=64)
    attn = TreeAttention(cfg).eval()
    x = torch.randn(2, 128, 32)
    with torch.no_grad():
        out = attn(x)
        C, H, dh = 32, 2, 16
        q, k, v = attn.c_attn(x).split(C, 2)
        q, k, v = [t.view(2, 128, H, dh).transpose(1, 2) for t in (q, k, v)]
        s = (q @ k.transpose(-1, -2)) / dh ** 0.5
        s = s.masked_fill(~torch.tril(torch.ones(128, 128, dtype=torch.bool)), float('-inf'))
        ref = attn.c_proj((torch.softmax(s, -1) @ v).transpose(1, 2).reshape(2, 128, C))
    err = (out - ref).abs().max().item()
    print('  max difference to dense attention: %.2e' % err)
    assert err < 1e-4, err


def test_causality():
    """Changing a token must not change the output at any EARLIER position (no leak from the future)."""
    for kw in [dict(), dict(gate=True), dict(beam_mode='pred', global_pool=True, gate=True)]:
        torch.manual_seed(1)
        model = GPT(tiny(**kw)).eval()
        idx = torch.randint(0, 64, (2, 128))
        idx2 = idx.clone()
        pos = 70
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
    torch.manual_seed(2)
    model = GPT(tiny(beam_mode='pred', global_pool=True, gate=True)).train()
    idx = torch.randint(0, 64, (2, 128))
    tgt = torch.randint(0, 64, (2, 128))
    _, loss, st = model(idx, tgt)
    loss.backward()
    print('  lm %.3f route %.3f pred %.3f keys tree %.1f total %.1f' % (
        float(st['lm']), st['route'], st['pred'], st['keys_tree'], st['keys_total']))
    assert torch.isfinite(loss)
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
    assert model.blocks[0].attn.c_attn.weight.grad.abs().sum() > 0
    assert st['route'] > 0 and st['pred'] > 0


if __name__ == '__main__':
    failed = 0
    for fn in [test_all_keys_matches_dense, test_causality, test_backward_and_losses]:
        try:
            fn()
            print('PASS', fn.__name__)
        except Exception:
            failed += 1
            print('FAIL', fn.__name__)
            traceback.print_exc()
    print('ALL TESTS PASSED' if failed == 0 else '%d TEST(S) FAILED' % failed)
    sys.exit(1 if failed else 0)
