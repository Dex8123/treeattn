"""Build data/long from data/tinystories: blocks of B tokens (aligned), a fraction of them recall blocks.
train.bin = mixed blocks, val.bin = plain text blocks only (so the validation loss is a clean language-model loss).
Both our model and nanoGPT train on exactly these files; nothing else differs between them.
"""
import argparse
import os
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn.recall import make_recall_block

p = argparse.ArgumentParser()
p.add_argument('--src', default='data/tinystories')
p.add_argument('--out', default='data/long')
p.add_argument('--B', type=int, default=4096)
p.add_argument('--p_recall', type=float, default=0.3)
p.add_argument('--n_needles', type=int, default=16)
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
V = pickle.load(open(os.path.join(a.src, 'meta.pkl'), 'rb'))['vocab_size']
rng = np.random.default_rng(a.seed)
for split in ('train', 'val'):
    data = np.fromfile(os.path.join(a.src, split + '.bin'), dtype=np.uint16)
    nb = len(data) // a.B
    blocks = data[:nb * a.B].reshape(nb, a.B)
    out, n_rec = [], 0
    for i in range(nb):
        if split == 'train' and rng.random() < a.p_recall:
            out.append(make_recall_block(blocks[i].astype(np.int64), V, rng, a.n_needles)[0].astype(np.uint16))
            n_rec += 1
        else:
            out.append(blocks[i])
    np.concatenate(out).astype(np.uint16).tofile(os.path.join(a.out, split + '.bin'))
    print('%s: %d blocks of %d tokens (%d recall blocks)' % (split, nb, a.B, n_rec))
shutil.copy(os.path.join(a.src, 'meta.pkl'), os.path.join(a.out, 'meta.pkl'))
