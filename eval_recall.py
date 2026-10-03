"""Recall accuracy by distance, for our model or for nanoGPT (nanoGPT's own model code is imported, not copied).

  python scripts/eval_recall.py --kind ours    --ckpt runs/long_seg/ckpt.pt
  python scripts/eval_recall.py --kind nanogpt --ckpt nanoGPT/out-long-base/ckpt.pt --nanogpt_dir nanoGPT
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn.recall import make_recall_block

p = argparse.ArgumentParser()
p.add_argument('--kind', required=True, choices=['ours', 'nanogpt'])
p.add_argument('--ckpt', required=True)
p.add_argument('--nanogpt_dir', default='nanoGPT')
p.add_argument('--data_dir', default='data/long')
p.add_argument('--B', type=int, default=4096)
p.add_argument('--n_docs', type=int, default=64)
p.add_argument('--n_needles', type=int, default=16)
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
import pickle
V = pickle.load(open(os.path.join(a.data_dir, 'meta.pkl'), 'rb'))['vocab_size']
ck = torch.load(a.ckpt, map_location='cpu')
if a.kind == 'ours':
    from treeattn.config import Config
    from treeattn.model import GPT
    model = GPT(Config(**ck['cfg']))
    model.load_state_dict(ck['model'])
else:
    sys.path.insert(0, a.nanogpt_dir)
    from model import GPT as NanoGPT, GPTConfig
    model = NanoGPT(GPTConfig(**ck['model_args']))
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()})
model.to(dev).eval()
val = np.fromfile(os.path.join(a.data_dir, 'val.bin'), dtype=np.uint16)
blocks = val[:(len(val) // a.B) * a.B].reshape(-1, a.B)
rng = np.random.default_rng(123)
hit, cnt = np.zeros(4), np.zeros(4)
with torch.no_grad():
    for start in range(0, a.n_docs, 2):
        docs = [make_recall_block(blocks[(start + j) % len(blocks)].astype(np.int64), V, rng, a.n_needles) for j in range(2)]
        x = torch.from_numpy(np.stack([d[0] for d in docs])).to(dev)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            logits = model(x)[0] if a.kind == 'ours' else model(x, x)[0]
        for j, (_, qpos, npos, vals) in enumerate(docs):
            pred = logits[j, torch.from_numpy(qpos).to(dev)].argmax(-1).cpu().numpy()
            bucket = np.minimum((qpos - npos) * 4 // a.B, 3)
            for b_, ok in zip(bucket, pred == vals):
                cnt[b_] += 1
                hit[b_] += float(ok)
print('%s recall over %d documents x %d queries' % (a.kind, a.n_docs, a.n_needles))
print('overall accuracy %.3f   (guessing one of the stored values = %.3f)' % (hit.sum() / cnt.sum(), 1.0 / a.n_needles))
print('by needle distance, near -> far quartiles of %d tokens: %s' % (
    a.B, ' '.join('%.3f' % (h / c) if c else '  -  ' for h, c in zip(hit, cnt))))
