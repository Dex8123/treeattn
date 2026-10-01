import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn.config import Config
from treeattn.data import get_batch
from treeattn.model import GPT

p = argparse.ArgumentParser()
p.add_argument('--data_dir', default='data/tinystories')
p.add_argument('--out_dir', required=True)
p.add_argument('--max_steps', type=int, default=3000)
p.add_argument('--batch_size', type=int, default=8)
p.add_argument('--accum', type=int, default=2)
p.add_argument('--lr', type=float, default=6e-4)
p.add_argument('--min_lr', type=float, default=6e-5)
p.add_argument('--warmup', type=int, default=100)
p.add_argument('--wd', type=float, default=0.1)
p.add_argument('--eval_interval', type=int, default=200)
p.add_argument('--eval_iters', type=int, default=20)
p.add_argument('--log_interval', type=int, default=10)
p.add_argument('--seed', type=int, default=1337)
p.add_argument('--seq_len', type=int, default=512)
p.add_argument('--n_layer', type=int, default=6)
p.add_argument('--n_head', type=int, default=6)
p.add_argument('--n_embd', type=int, default=384)
p.add_argument('--tree_block', type=int, default=64)
p.add_argument('--wmax', type=int, default=8)
p.add_argument('--beam', type=int, default=4)
p.add_argument('--beam_mode', default='fixed')
p.add_argument('--global_pool', type=int, default=0)
p.add_argument('--gate', type=int, default=0)
p.add_argument('--route_weight', type=float, default=0.05)
p.add_argument('--pred_weight', type=float, default=0.1)
p.add_argument('--kmeans_iters', type=int, default=3)
a = p.parse_args()

os.makedirs(a.out_dir, exist_ok=True)
torch.manual_seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = dev == 'cuda'
vocab = pickle.load(open(os.path.join(a.data_dir, 'meta.pkl'), 'rb'))['vocab_size']
cfg = Config(vocab_size=vocab, seq_len=a.seq_len, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd,
             tree_block=a.tree_block, wmax=a.wmax, beam=a.beam, beam_mode=a.beam_mode,
             global_pool=bool(a.global_pool), gate=bool(a.gate), route_weight=a.route_weight,
             pred_weight=a.pred_weight, kmeans_iters=a.kmeans_iters)
json.dump({'config': cfg.to_dict(), 'args': vars(a)}, open(os.path.join(a.out_dir, 'config.json'), 'w'), indent=1)
model = GPT(cfg).to(dev)
print('device', dev, '| params %.2fM' % (sum(p_.numel() for p_ in model.parameters()) / 1e6), flush=True)

decay = [p_ for p_ in model.parameters() if p_.requires_grad and p_.dim() >= 2]
nodecay = [p_ for p_ in model.parameters() if p_.requires_grad and p_.dim() < 2]
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd}, {'params': nodecay, 'weight_decay': 0.0}],
                        lr=a.lr, betas=(0.9, 0.95))
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
train_path = os.path.join(a.data_dir, 'train.bin')
val_path = os.path.join(a.data_dir, 'val.bin')
rng = np.random.default_rng(a.seed)


def get_lr(step):
    if step < a.warmup:
        return a.lr * (step + 1) / (a.warmup + 1)
    r = (step - a.warmup) / max(1, a.max_steps - a.warmup)
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * min(1.0, r))) * (a.lr - a.min_lr)


@torch.no_grad()
def estimate():
    model.eval()
    r = np.random.default_rng(999)
    tot = 0.0
    for _ in range(a.eval_iters):
        x, y = get_batch(val_path, a.batch_size, a.seq_len, dev, r)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            _, _, st = model(x, y)
        tot += float(st['lm'])
    model.train()
    return tot / a.eval_iters


cols = ['step', 'tokens', 'lr', 'lm', 'route', 'pred', 'keys_tree', 'keys_total', 'sec_per_step', 'val_loss']
fcsv = open(os.path.join(a.out_dir, 'metrics.csv'), 'w', newline='')
wr = csv.DictWriter(fcsv, fieldnames=cols)
wr.writeheader()
tok_per_step = a.batch_size * a.accum * a.seq_len
model.train()
for step in range(a.max_steps + 1):
    lr = get_lr(step)
    for g in opt.param_groups:
        g['lr'] = lr
    if step % a.eval_interval == 0 or step == a.max_steps:
        vl = estimate()
        print('step %d | val loss %.4f' % (step, vl), flush=True)
        wr.writerow({'step': step, 'tokens': step * tok_per_step, 'val_loss': round(vl, 5)})
        fcsv.flush()
    if step == a.max_steps:
        break
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    agg = defaultdict(float)
    for _ in range(a.accum):
        x, y = get_batch(train_path, a.batch_size, a.seq_len, dev, rng)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            _, loss, st = model(x, y)
        scaler.scale(loss / a.accum).backward()
        for k_, v_ in st.items():
            agg[k_] += float(v_) / a.accum
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    dt = time.time() - t0
    if step % a.log_interval == 0:
        print('step %d | lm %.4f | route %.4f | pred %.4f | keys tree %.1f total %.1f | %.2fs/step | lr %.2e' % (
            step, agg['lm'], agg['route'], agg['pred'], agg['keys_tree'], agg['keys_total'], dt, lr), flush=True)
        wr.writerow({'step': step, 'tokens': (step + 1) * tok_per_step, 'lr': round(lr, 8),
                     'lm': round(agg['lm'], 5), 'route': round(agg['route'], 5), 'pred': round(agg['pred'], 5),
                     'keys_tree': round(agg['keys_tree'], 2), 'keys_total': round(agg['keys_total'], 2),
                     'sec_per_step': round(dt, 3)})
        fcsv.flush()
torch.save({'model': model.state_dict(), 'cfg': cfg.to_dict()}, os.path.join(a.out_dir, 'ckpt.pt'))
print('DONE', flush=True)
