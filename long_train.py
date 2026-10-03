"""Beta 2: long-context training for treeattn (seg) against a dense baseline, plus a non-local recall test.

New file only: nothing in the repo is replaced. It patches the gate so it only applies at tree levels >= gate_min_level.

  python scripts/long_train.py --selftest                      (checks, memory, timing; no training)
  python scripts/long_train.py --attn dense --out_dir runs/long_dense
  python scripts/long_train.py --attn seg   --out_dir runs/long_seg --gate 1 --gate_min_level 6
"""
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn import seg as segmod
from treeattn.config import Config
from treeattn.data import get_batch
from treeattn.model import GPT

# ---------------------------------------------------------------- gate only at the top tree levels
_GATE_MIN = [0]


@torch.no_grad()
def level_sums_min(kidx, Lm, gate, gate_q):
    """Same as seg.level_sums, but the mean-log gate is applied only at levels >= _GATE_MIN[0].
    Parents are still sums of the ungated children."""
    b, H, T, dh = kidx.shape
    raw = kidx
    lg = torch.log(kidx.abs() + 1e-6) if gate else None
    out = {}
    for l in range(1, Lm + 1):
        raw = raw.reshape(b, H, T >> l, 2, dh).sum(3)
        if gate:
            lg = lg.reshape(b, H, T >> l, 2, dh).sum(3)
            out[l] = segmod.gate_apply(raw, lg, 1 << l, gate_q) if l >= _GATE_MIN[0] else raw
        else:
            out[l] = raw
    return out


segmod.level_sums = level_sums_min


# ---------------------------------------------------------------- dense baseline (same model, dense causal attention)
class DenseAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg.n_embd
        self.H = cfg.n_head
        self.dh = C // cfg.n_head
        self.c_attn = nn.Linear(C, 3 * C, bias=False)
        self.c_proj = nn.Linear(C, C, bias=False)
        self.aux = {}

    def forward(self, x):
        b, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = [t.view(b, T, self.H, self.dh).transpose(1, 2) for t in (q, k, v)]
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(o.transpose(1, 2).reshape(b, T, C))


def make_model(cfg, kind):
    model = GPT(cfg)
    if kind == 'dense':
        for blk in model.blocks:
            blk.attn = DenseAttention(cfg)
        model.apply(GPT._init)
        for n, p in model.named_parameters():
            if n.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
    return model


def make_cfg(a, V):
    return Config(vocab_size=V, seq_len=a.seq_len, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd, attn='seg',
                  beam_mode='pred', global_pool=True, gate=bool(a.gate), gate_min_level=a.gate_min_level, local=3, chunk=64, neighbors=True)


# ---------------------------------------------------------------- synthetic recall task
# A window of ordinary text (rare token ids remapped away) holds P needles "MARK key value" at random places.
# The window ends with "QRY key_j"; the model must output value_j. Loss only on that last position.
NK = 500  # size of the key pool and of the value pool


def make_recall(V, L, nb, rng, dev, path, P=6, Q=1, filler=0):
    """P needles \"MARK key value\" in a text window; the window ends with Q bare queries \"QRY key\" (no answer
    in the input). Labels (value) sit only at the Q key positions, everything else is -100. pos: (nb, Q) needle starts."""
    KEY0, VAL0, MARK, QRY = V - 1100, V - 600, V - 100, V - 99
    assert 1 <= Q <= P
    x, _ = get_batch(path, nb, L, dev, rng)
    x = x.clone()
    x = torch.where(x >= V - 1100, x - 1100, x)
    xn = x.cpu().numpy().copy()
    if filler:
        xn[:] = 2  # constant filler token instead of text (as in the MQAR benchmark)
    yn = np.full((nb, L), -100, dtype=np.int64)
    pos = np.zeros((nb, Q), dtype=np.int64)
    n_slots = (L - 2 * Q - 4) // 4
    for i in range(nb):
        slots = rng.choice(n_slots, P, replace=False)
        keys = rng.choice(NK, P, replace=False)
        vals = rng.integers(0, NK, P)
        for s_, kk, vv in zip(slots, keys, vals):
            p_ = int(s_) * 4
            xn[i, p_], xn[i, p_ + 1], xn[i, p_ + 2] = MARK, KEY0 + int(kk), VAL0 + int(vv)
        qs = rng.choice(P, Q, replace=False)
        for jq, j in enumerate(qs):
            b_ = L - 2 * Q + 2 * jq
            xn[i, b_], xn[i, b_ + 1] = QRY, KEY0 + int(keys[j])
            yn[i, b_ + 1] = VAL0 + int(vals[j])
            pos[i, jq] = int(slots[j]) * 4
    return torch.from_numpy(xn).to(dev), torch.from_numpy(yn).to(dev), pos


# ---------------------------------------------------------------- args
p = argparse.ArgumentParser()
p.add_argument('--attn', default='seg', choices=['seg', 'dense'])
p.add_argument('--data_dir', default='data/tinystories')
p.add_argument('--out_dir', default='runs/long')
p.add_argument('--selftest', action='store_true')
p.add_argument('--seq_len', type=int, default=4096)
p.add_argument('--n_layer', type=int, default=6)
p.add_argument('--n_head', type=int, default=6)
p.add_argument('--n_embd', type=int, default=384)
p.add_argument('--max_steps', type=int, default=1500)
p.add_argument('--max_minutes', type=float, default=600.0)
p.add_argument('--batch_size', type=int, default=2)
p.add_argument('--accum', type=int, default=4)
p.add_argument('--lr', type=float, default=6e-4)
p.add_argument('--min_lr', type=float, default=6e-5)
p.add_argument('--warmup', type=int, default=100)
p.add_argument('--wd', type=float, default=0.1)
p.add_argument('--gate', type=int, default=1)
p.add_argument('--gate_min_level', type=int, default=6)
p.add_argument('--p_recall', type=float, default=0.25)
p.add_argument('--eval_interval', type=int, default=250)
p.add_argument('--eval_iters', type=int, default=20)
p.add_argument('--recall_eval', type=int, default=64)
p.add_argument('--needles', type=int, default=6)
p.add_argument('--queries', type=int, default=1)
p.add_argument('--filler', type=int, default=0)
p.add_argument('--nk', type=int, default=500)  # size of the key pool and of the value pool (<= 500)
p.add_argument('--lens', type=str, default='')  # e.g. 256 or 128,256 (recall window lengths, each <= seq_len)
p.add_argument('--log_interval', type=int, default=10)
p.add_argument('--seed', type=int, default=1337)
a = p.parse_args()

_GATE_MIN[0] = a.gate_min_level
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = dev == 'cuda'
V = pickle.load(open(os.path.join(a.data_dir, 'meta.pkl'), 'rb'))['vocab_size']
train_path = os.path.join(a.data_dir, 'train.bin')
val_path = os.path.join(a.data_dir, 'val.bin')
LENS = [int(v) for v in a.lens.split(',')] if a.lens else [L for L in (512, 1024, 2048, 4096) if L <= a.seq_len]
assert all(L <= a.seq_len for L in LENS), '--lens entries must be <= --seq_len'
assert 1 <= a.nk <= 500 and a.queries <= a.needles <= a.nk, 'need queries <= needles <= nk <= 500'
NK = a.nk


def sync():
    if dev == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------- selftest
def selftest():
    cfg = make_cfg(a, V)
    torch.manual_seed(0)
    ok = True
    T = a.seq_len
    for kind in ('seg', 'dense'):
        m = make_model(cfg, kind).to(dev).eval()
        x = torch.randint(0, V - 1200, (1, T), device=dev)
        pos = T // 2 + 37
        x2 = x.clone()
        x2[0, pos] = (x2[0, pos] + 1) % (V - 1200)
        with torch.no_grad():
            l1, _, _ = m(x)
            l2, _, _ = m(x2)
        before = (l1[:, :pos] - l2[:, :pos]).abs().max().item()
        after = (l1[:, pos:] - l2[:, pos:]).abs().max().item()
        good = before < 1e-4 and after > 0
        ok &= good
        print('[%s] no-leak at T=%d: change before edit %.2e (must be ~0), after edit %.2e -> %s' % (
            kind, T, before, after, 'PASS' if good else 'FAIL'), flush=True)
        del m
    if dev == 'cuda':
        m = make_model(cfg, 'seg').to(dev)
        q = torch.randn(1, a.n_head, T, a.n_embd // a.n_head, device=dev)
        k = torch.randn_like(q)
        lhat = torch.full((1, a.n_head, T), 4, dtype=torch.long, device=dev)
        segmod.select_seg(q, k, lhat, cfg)
        sync()
        t0 = time.time()
        for _ in range(3):
            segmod.select_seg(q, k, lhat, cfg)
        sync()
        sel = (time.time() - t0) / 3
        x = torch.randint(0, V - 1200, (1, T), device=dev)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            m(x)
            sync()
            t0 = time.time()
            for _ in range(3):
                m(x)
            sync()
        fwd = (time.time() - t0) / 3
        print('timing at T=%d, batch 1: selection %.0f ms per layer (x%d layers = %.0f ms) of a %.0f ms forward' % (
            T, sel * 1e3, a.n_layer, sel * a.n_layer * 1e3, fwd * 1e3), flush=True)
        torch.cuda.reset_peak_memory_stats()
        m.train()
        xb, yb = get_batch(train_path, a.batch_size, T, dev, np.random.default_rng(0))
        t0 = time.time()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, loss, _ = m(xb, yb)
        loss.backward()
        sync()
        print('one train micro-batch (batch %d, T=%d): %.1f s, peak memory %.1f GB (T4 has ~15)' % (
            a.batch_size, T, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    print('SELFTEST', 'PASS' if ok else 'FAIL', flush=True)
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model):
    model.eval()
    r = np.random.default_rng(999)
    tot = 0.0
    for _ in range(a.eval_iters):
        x, y = get_batch(val_path, a.batch_size, a.seq_len, dev, r)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            _, _, st = model(x, y)
        tot += float(st['lm'])
    res = {'val_loss': tot / a.eval_iters, 'recall': {}}
    for L in LENS:
        rr = np.random.default_rng(1000 + L)
        hit, cnt = np.zeros(4), np.zeros(4)
        n_done = 0
        while n_done < a.recall_eval:
            nb = 2
            x, y, pos = make_recall(V, L, nb, rr, dev, val_path, a.needles, a.queries, a.filler)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits, _, _ = model(x)
            kp = L - 2 * a.queries + 1 + 2 * np.arange(a.queries)  # positions of the query keys
            pred = logits[:, torch.from_numpy(kp).to(dev)].argmax(-1).cpu().numpy()
            tgt = y[:, torch.from_numpy(kp).to(dev)].cpu().numpy()
            bucket = np.minimum((kp[None, :] - pos) * 4 // L, 3)  # 0 = needle closest to the query, 3 = farthest
            for i in range(nb):
                for j in range(a.queries):
                    cnt[bucket[i, j]] += 1
                    hit[bucket[i, j]] += float(pred[i, j] == tgt[i, j])
            n_done += nb
        res['recall'][L] = {'all': float(hit.sum() / max(cnt.sum(), 1)),
                            'by_distance_quartile': [float(h / c) if c else None for h, c in zip(hit, cnt)]}
    model.train()
    return res


def show(step, res):
    print('step %d | val loss %.4f' % (step, res['val_loss']), flush=True)
    for L, r in res['recall'].items():
        q = ' '.join('%s' % ('%.2f' % v if v is not None else ' - ') for v in r['by_distance_quartile'])
        print('   recall L=%d: acc %.2f | near->far quartiles: %s   (guess-among-stored-values = %.3f)' % (L, r['all'], q, 1.0 / a.needles),
              flush=True)


# ---------------------------------------------------------------- training
def train():
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    cfg = make_cfg(a, V)
    json.dump({'config': cfg.to_dict(), 'args': vars(a)}, open(os.path.join(a.out_dir, 'config.json'), 'w'), indent=1)
    model = make_model(cfg, a.attn).to(dev)
    print('device', dev, '| attn', a.attn, '| params %.2fM' % (sum(p_.numel() for p_ in model.parameters()) / 1e6),
          '| seq_len', a.seq_len, '| gate', a.gate, 'from level', a.gate_min_level, flush=True)
    decay = [p_ for p_ in model.parameters() if p_.requires_grad and p_.dim() >= 2]
    nodecay = [p_ for p_ in model.parameters() if p_.requires_grad and p_.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd}, {'params': nodecay, 'weight_decay': 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = np.random.default_rng(a.seed)

    def get_lr(step):
        if step < a.warmup:
            return a.lr * (step + 1) / (a.warmup + 1)
        r = (step - a.warmup) / max(1, a.max_steps - a.warmup)
        return a.min_lr + 0.5 * (1 + math.cos(math.pi * min(1.0, r))) * (a.lr - a.min_lr)

    fcsv = open(os.path.join(a.out_dir, 'metrics.csv'), 'w', newline='')
    cols = ['step', 'lm_text', 'lm_recall', 'sec_per_step', 'val_loss', 'recall_4096', 'recall_2048']
    wr = csv.DictWriter(fcsv, fieldnames=cols)
    wr.writeheader()
    tok_mb = a.batch_size * a.seq_len
    t_start = time.time()
    model.train()
    for step in range(a.max_steps + 1):
        for g in opt.param_groups:
            g['lr'] = get_lr(step)
        out_of_time = (time.time() - t_start) > a.max_minutes * 60
        if step % a.eval_interval == 0 or step == a.max_steps or out_of_time:
            res = evaluate(model)
            show(step, res)
            wr.writerow({'step': step, 'val_loss': round(res['val_loss'], 5),
                         'recall_4096': res['recall'].get(4096, {}).get('all'),
                         'recall_2048': res['recall'].get(2048, {}).get('all')})
            fcsv.flush()
            json.dump(res, open(os.path.join(a.out_dir, 'last_eval.json'), 'w'), indent=1)
            torch.save({'model': model.state_dict(), 'cfg': cfg.to_dict()}, os.path.join(a.out_dir, 'ckpt.pt'))
        if step == a.max_steps or out_of_time:
            if out_of_time:
                print('stopped by --max_minutes at step %d (learning rate was not fully decayed)' % step, flush=True)
            break
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        agg, n_agg = defaultdict(float), defaultdict(int)
        for _ in range(a.accum):
            if rng.random() < a.p_recall:
                L = int(rng.choice(LENS))
                x, y, _ = make_recall(V, L, max(1, tok_mb // L), rng, dev, train_path, a.needles, a.queries, a.filler)
                kind = 'lm_recall'
            else:
                x, y = get_batch(train_path, a.batch_size, a.seq_len, dev, rng)
                kind = 'lm_text'
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                _, loss, st = model(x, y)
            scaler.scale(loss / a.accum).backward()
            agg[kind] += float(st['lm'])
            n_agg[kind] += 1
            for k_ in ('keys_total', 'keys_tree', 'keys_nb', 'pred'):
                if k_ in st:
                    agg[k_] += float(st[k_]) / a.accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        dt = time.time() - t0
        if step % a.log_interval == 0:
            parts = ['%s %.3f' % (k_, agg[k_] / n_agg[k_]) for k_ in ('lm_text', 'lm_recall') if n_agg[k_]]
            parts += ['%s %.2f' % (k_, agg[k_]) for k_ in ('keys_total', 'keys_tree', 'keys_nb', 'pred') if k_ in agg]
            print('step %d | %s | %.2fs/step | lr %.2e' % (step, ' | '.join(parts), dt, get_lr(step)), flush=True)
        wr.writerow({'step': step, 'lm_text': round(agg['lm_text'] / max(n_agg['lm_text'], 1), 5),
                     'lm_recall': round(agg['lm_recall'] / max(n_agg['lm_recall'], 1), 5), 'sec_per_step': round(dt, 3)})
    print('DONE', flush=True)


if a.selftest:
    selftest()
else:
    train()
