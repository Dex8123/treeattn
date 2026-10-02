import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn.attention import TreeAttention
from treeattn.attention_dyn import DynAttention
from treeattn.config import Config

assert torch.cuda.is_available(), 'this benchmark needs a GPU'
dev = 'cuda'
H, dh = 6, 64
LENGTHS = [512, 1024, 2048, 4096]


def timeit(fn, iters=3):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


print('One attention layer, 6 heads, head size 64, fp16, forward only, batch 1.')
print('dense = PyTorch scaled_dot_product_attention (flash-style kernel); block = beta 1; dyn = round 4 dynamic tree.')
rows = []
for T in LENGTHS:
    q, k, v = [torch.randn(1, H, T, dh, device=dev, dtype=torch.float16) for _ in range(3)]
    res = {'T': T}
    try:
        with torch.no_grad():
            res['dense'] = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
            cfg_b = Config(seq_len=T, n_head=H, n_embd=H * dh, tree_block=64, beam=4)
            blk = TreeAttention(cfg_b).to(dev).eval()
            res['block'] = timeit(lambda: blk.sparse_attend(q, k, v))
            del blk
            cfg_d = Config(seq_len=T, n_head=H, n_embd=H * dh, attn='dyn', chunk=64, beam=4, beam_mode='fixed', local=3)
            dyn = DynAttention(cfg_d).to(dev).eval()
            res['dyn'] = timeit(lambda: dyn.sparse_attend(q, k, v), iters=2)
            del dyn
    except torch.cuda.OutOfMemoryError:
        print('T=%d: out of memory' % T, flush=True)
    torch.cuda.empty_cache()
    rows.append(res)
    print('T=%5d | dense %8.2f ms | block %8.2f ms | dyn %9.2f ms' % (
        T, res.get('dense', float('nan')), res.get('block', float('nan')), res.get('dyn', float('nan'))), flush=True)

print()
print('Growth when the length doubles (x4 would be quadratic, x2 linear). Small lengths are dominated by fixed overhead.')
for a, b in zip(rows[:-1], rows[1:]):
    parts = []
    for name in ('dense', 'block', 'dyn'):
        if name in a and name in b:
            parts.append('%s x%.2f' % (name, b[name] / a[name]))
    print('%5d -> %5d: %s' % (a['T'], b['T'], ' | '.join(parts)))
