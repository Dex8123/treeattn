"""Router-only probe: does our key selection pick up a planted needle, with no training involved?

For each context length T it builds random queries and keys (every logit q.k/sqrt(dh) ~ N(0,1)), plants ONE needle
key at a random position whose logit for the LAST query is `margin` above the noise, and asks two questions:
  - dense attention: how much softmax weight would the last query put on the needle?
  - our selection (the same functions the model uses: select_seg_fast + dedupe_fast): is the needle among the keys
    that are actually kept for the last query?
Random keys means no co-adaptation: a trained model may do better (or worse). This isolates the routing step.

  python scripts/router_probe.py                      # T = 512, 4096, 16384
  python scripts/router_probe.py --gate 0             # without the mean-log gate
  python scripts/router_probe.py --budget 8           # larger leaf budget
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn.config import Config
from treeattn.fastseg import dedupe_fast, select_seg_fast

p = argparse.ArgumentParser()
p.add_argument('--T', type=str, default='512,4096,16384')
p.add_argument('--margins', type=str, default='0,4,6,8,10,12,14,16,20')
p.add_argument('--budget', type=int, default=4)       # leaf groups per query (classes the head picks from: 2,3,4,6,8)
p.add_argument('--gate', type=int, default=1)
p.add_argument('--gate_min_level', type=int, default=6)
p.add_argument('--trials', type=int, default=96)
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
if not torch.cuda.is_available():
    sys.exit('this probe needs a GPU')
dev = 'cuda'
H, dh = 6, 64


@torch.no_grad()
def one_call(T, m, g, cfg, b):
    q = torch.randn(b, H, T, dh, device=dev, generator=g)
    k = torch.randn(b, H, T, dh, device=dev, generator=g)
    pos = torch.randint(8, T - 16, (b, H), device=dev, generator=g)       # needle position, away from sink and local keys
    qT = q[:, :, T - 1, :]
    if m > 0:
        needle = m * qT / qT.norm(dim=-1, keepdim=True)                    # logit of the needle for the last query = m
        k.scatter_(2, pos[..., None, None].expand(b, H, 1, dh), needle[:, :, None, :])
    s = torch.einsum('bhd,bhtd->bht', qT, k[:, :, :T - 1]) / math.sqrt(dh)
    w_dense = torch.softmax(s, -1).gather(-1, pos.unsqueeze(-1)).squeeze(-1)   # (b,H)

    lhat = torch.full((b, H, T), a.budget, dtype=torch.long, device=dev)
    sel = select_seg_fast(q, k, lhat, cfg)
    tree_pos, nb_pos, leaf_ok = sel['tree_pos'], sel['nb_pos'], sel['leaf_ok']
    wl = leaf_ok.shape[-1]
    t = torch.arange(T, device=dev)
    Lw = max(cfg.local, 3)
    st = torch.cat([torch.zeros(T, 1, dtype=torch.long, device=dev), t.unsqueeze(1) - torch.arange(Lw, device=dev).unsqueeze(0)], 1)
    st = st.masked_fill(st < 0, -1).view(1, 1, T, 1 + Lw).expand(b, H, T, 1 + Lw)
    cand = torch.cat([tree_pos, nb_pos, st], -1)
    used = leaf_ok & (torch.arange(wl, device=dev) < sel['n_tree'].unsqueeze(-1))
    tv_used = torch.cat([used, used], -1) & (tree_pos >= 0)
    rest = torch.cat([nb_pos >= 0, st >= 0], -1)
    last = T - 1
    cand_l = cand[:, :, last:last + 1, :]
    valid_l = torch.cat([tv_used, rest], -1)[:, :, last:last + 1, :]
    keep = dedupe_fast(cand_l, valid_l)[:, :, 0, :]                          # (b,H,Ncand)
    final = torch.where(keep, cand_l[:, :, 0, :], torch.full_like(cand_l[:, :, 0, :], -1))
    found = (final == pos.unsqueeze(-1)).any(-1)
    tree_only = ((torch.where(tv_used[:, :, last, :], tree_pos[:, :, last, :], torch.full_like(tree_pos[:, :, last, :], -1))
                  == pos.unsqueeze(-1)).any(-1))
    return w_dense.flatten().cpu(), found.flatten().float().cpu(), tree_only.flatten().float().cpu(), keep.sum(-1).flatten().float().cpu()


print('gate %d (from level %d) | budget %d leaf groups | %d trials per cell | random keys, one planted needle' % (
    a.gate, a.gate_min_level, a.budget, a.trials))
g = torch.Generator(device=dev)
for T in [int(v) for v in a.T.split(',')]:
    cfg = Config(vocab_size=8192, seq_len=T, n_layer=1, n_head=H, n_embd=H * dh, attn='seg', beam_mode='pred', global_pool=True,
                 gate=bool(a.gate), gate_min_level=a.gate_min_level, local=3, chunk=64, neighbors=True)
    b = min(4, max(1, 8192 // T))
    calls = max(1, math.ceil(a.trials / (b * H)))
    print('\nT = %d   (chance that a random position is read: about %.3f)' % (T, 1.0 / T))
    print(' margin | dense weight on needle | ours: needle kept | ours: via tree | keys kept')
    d50 = r90 = None
    for m in [float(v) for v in a.margins.split(',')]:
        g.manual_seed(a.seed + int(m * 10))
        ws, fs, ts, ks = [], [], [], []
        for _ in range(calls):
            w_, f_, t_, k_ = one_call(T, m, g, cfg, b)
            ws.append(w_); fs.append(f_); ts.append(t_); ks.append(k_)
        w, f, tr, kk = [torch.cat(x).mean().item() for x in (ws, fs, ts, ks)]
        n = calls * b * H
        print(' %6.1f | %22.3f | %12.3f (+-%.2f) | %14.3f | %9.1f' % (m, w, f, math.sqrt(max(f * (1 - f), 1e-9) / n), tr, kk))
        if d50 is None and w >= 0.5: d50 = m
        if r90 is None and f >= 0.9: r90 = m
    print(' -> dense puts >= 50%% of its weight on the needle from margin %s; ours keeps it in >= 90%% of trials from margin %s' % (
        d50, r90))
