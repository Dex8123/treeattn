import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn.config import Config
from treeattn.engine import SegState, seg_attend

assert torch.cuda.is_available(), 'this benchmark needs a GPU'
dev = 'cuda'
H, dh = 6, 64
LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def timeit(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


print('Cost of ONE decoding step of ONE attention layer at a given context length (6 heads, head size 64, fp32).')
print('dense = cached keys/values + scaled_dot_product_attention; seg = segment-tree search + neighbours + exact attention.')
rows = []
for T in LENGTHS:
    try:
        cfg = Config(seq_len=T, n_head=H, n_embd=H * dh, attn='seg', beam=4, beam_mode='fixed', local=3)
        K, V = torch.randn(1, H, T, dh, device=dev), torch.randn(1, H, T, dh, device=dev)
        q = torch.randn(1, H, 1, dh, device=dev)
        st = SegState(cfg, T, H, dh, dev)
        st.fill(K, V)
        dense = timeit(lambda: F.scaled_dot_product_attention(q, K, V))
        seg = timeit(lambda: seg_attend(st, None, q, None, None, T - 1, update=False))
        keys = seg_attend(st, None, q, None, None, T - 1, update=False)[1]
        rows.append((T, dense, seg))
        print('context %6d | dense %7.3f ms (reads %6d keys) | seg %7.3f ms (reads %5.1f keys)' % (T, dense, T, seg, keys), flush=True)
        del st, K, V
    except torch.cuda.OutOfMemoryError:
        print('context %d: out of memory' % T, flush=True)
    torch.cuda.empty_cache()
print()
print('Growth when the context doubles (x2 = linear, x1 = flat):')
for a, b in zip(rows[:-1], rows[1:]):
    print('%6d -> %6d: dense x%.2f | seg x%.2f' % (a[0], b[0], b[1] / a[1], b[2] / a[2]))
