"""Checks that the fast seg path equals the old path, and times both. Run: python tests/test_fast.py"""
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn import fastseg
from treeattn.attention_dyn import DynAttention, dedupe
from treeattn.config import Config
from treeattn.fastseg import SegAttention, dedupe_fast, select_seg_fast
from treeattn.seg import select_seg

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
TOL = 1e-3


def tiny(**kw):
    base = dict(vocab_size=64, seq_len=128, n_layer=1, n_head=2, n_embd=128, attn='seg', beam_mode='pred',
                global_pool=True, wmax=8, local=3)
    base.update(kw)
    return Config(**base)


def test_dedupe():
    torch.manual_seed(0)
    cand = torch.randint(-1, 12, (3, 2, 50, 36), device=dev)
    valid = (torch.rand(3, 2, 50, 36, device=dev) < 0.8) & (cand >= 0)
    assert (dedupe(cand, valid) == dedupe_fast(cand, valid)).all()


def test_selection_same_as_seg():
    for gate in (False, True):
        torch.manual_seed(1)
        cfg = tiny(gate=gate)
        T = 256
        q, k = torch.randn(2, 2, T, 64, device=dev), torch.randn(2, 2, T, 64, device=dev)
        lhat = torch.randint(2, 9, (2, 2, T), device=dev)
        a, b = select_seg(q, k, lhat, cfg), select_seg_fast(q, k, lhat, cfg)
        same = ((a['tree_pos'] == b['tree_pos']).all(-1) & (a['nb_pos'] == b['nb_pos']).all(-1)).float().mean().item()
        print('  gate=%s: %.4f of queries select identical keys' % (gate, same))
        assert same > 0.995, same


def test_attention_same_as_old():
    for kw in [dict(), dict(gate=True), dict(beam_mode='fixed', global_pool=False)]:
        torch.manual_seed(2)
        cfg = tiny(**kw)
        old, new = DynAttention(cfg).to(dev), SegAttention(cfg).to(dev)
        new.load_state_dict(old.state_dict())
        x = torch.randn(2, 128, 128, device=dev)
        xo, xn = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        old.train(); new.train()
        yo, yn = old(xo), new(xn)
        w = torch.randn_like(yo)
        ((yo * w).sum() + sum(v for kk, v in old.aux.items() if kk == 'pred')).backward()
        ((yn * w).sum() + sum(v for kk, v in new.aux.items() if kk == 'pred')).backward()
        e_out = (yo - yn).abs().max().item()
        e_in = (xo.grad - xn.grad).abs().max().item()
        e_par = max((p1.grad - p2.grad).abs().max().item() for p1, p2 in zip(old.parameters(), new.parameters()))
        print('  %s: max difference output %.1e, input grad %.1e, weight grads %.1e' % (kw, e_out, e_in, e_par))
        assert e_out < TOL and e_in < TOL and e_par < TOL


def test_kernel_status():
    print('  Triton kernels in use:', fastseg._use_fast(torch.zeros(1, device=dev)))


def timing():
    if dev != 'cuda':
        print('  skipped (no GPU)')
        return
    cfg = tiny(seq_len=4096, n_head=6, n_embd=384, gate=True, gate_min_level=6)
    old, new = DynAttention(cfg).to(dev), SegAttention(cfg).to(dev)
    new.load_state_dict(old.state_dict())
    x = torch.randn(2, 4096, 384, device=dev, requires_grad=True)
    for name, m in (('old', old), ('new', new)):
        ts = []
        for _ in range(4):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y = m(x)
            y.float().sum().backward()
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        print('  %s: %.0f ms per layer, forward + backward, batch 2, T=4096' % (name, min(ts[1:]) * 1000))


if __name__ == '__main__':
    failed = 0
    for fn in [test_dedupe, test_selection_same_as_seg, test_attention_same_as_old, test_kernel_status, timing]:
        try:
            fn()
            print('PASS', fn.__name__)
        except Exception:
            failed += 1
            print('FAIL', fn.__name__)
            traceback.print_exc()
    print('ALL FAST TESTS PASSED' if failed == 0 else '%d TEST(S) FAILED' % failed)
    sys.exit(1 if failed else 0)
