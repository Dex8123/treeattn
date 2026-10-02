import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dyntree import select_keys
from .forest import gather_nodes
from .seg import select_seg


def dedupe(cand, valid):
    """Keep only the first copy of every key position among the valid candidates of a query."""
    Kc = cand.shape[-1]
    eq = (cand.unsqueeze(-1) == cand.unsqueeze(-2)) & valid.unsqueeze(-2)
    earlier = torch.tril(torch.ones(Kc, Kc, dtype=torch.bool, device=cand.device), diagonal=-1)
    return valid & ~(eq & earlier).any(-1)


class DynAttention(nn.Module):
    """Round 4 attention: dynamic single tree + neighbour search + budget head.

    Selection of keys (non-differentiable) is done by select_keys; the attention itself is exact softmax attention
    over the selected keys, so gradients flow through q, k, v of those keys only (as in any sparse attention)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C = cfg.n_embd
        self.H = cfg.n_head
        self.dh = C // cfg.n_head
        assert cfg.seq_len % cfg.chunk == 0 and (cfg.seq_len & (cfg.seq_len - 1)) == 0
        self.c_attn = nn.Linear(C, 3 * C, bias=False)
        self.c_proj = nn.Linear(C, C, bias=False)
        if cfg.beam_mode == 'pred':
            assert max(cfg.beam_classes) <= cfg.wmax
            fin = self.dh + 1 + self.H + (self.dh if cfg.global_pool else 0)
            self.pred = nn.Sequential(nn.Linear(fin, 64), nn.ReLU(), nn.Linear(64, len(cfg.beam_classes)))
            self.register_buffer('classes', torch.tensor(list(cfg.beam_classes)), persistent=False)
            self.register_buffer('head_eye', torch.eye(self.H), persistent=False)
        self.aux = {}

    def forward(self, x):
        b, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = [t.view(b, T, self.H, self.dh).transpose(1, 2) for t in (q, k, v)]
        out = self.sparse_attend(q, k, v)
        return self.c_proj(out.transpose(1, 2).reshape(b, T, C))

    def sparse_attend(self, q, k, v):
        cfg = self.cfg
        b, H, T, dh = q.shape
        dev = q.device
        scale = 1.0 / math.sqrt(dh)
        t = torch.arange(T, device=dev)
        self.aux = {}

        # budget: how many leaf groups (2 keys each) should this query get?
        plog = None
        if cfg.beam_mode == 'pred':
            qd = q.detach()
            feats = [qd,
                     torch.log(qd.float().norm(dim=-1, keepdim=True) + 1e-6).to(qd.dtype),
                     self.head_eye.to(qd.dtype)[None, :, None, :].expand(b, H, T, H)]
            if cfg.global_pool:
                kd = k.detach().float()
                prefix_mean = (kd.cumsum(2) - kd) / t.clamp(min=1).view(1, 1, T, 1).float()
                feats.append(prefix_mean.to(qd.dtype))
            plog = self.pred(torch.cat(feats, -1))
            lhat = self.classes[plog.argmax(-1).detach()]
        else:
            lhat = torch.full((b, H, T), cfg.beam, dtype=torch.long, device=dev)

        select = select_seg if cfg.attn == 'seg' else select_keys
        sel = select(q.detach(), k.detach(), lhat, cfg)
        tree_pos, nb_pos, leaf_ok = sel['tree_pos'], sel['nb_pos'], sel['leaf_ok']
        wl = leaf_ok.shape[-1]
        NBK = nb_pos.shape[-1]

        Lw = max(cfg.local, 3)
        st = torch.cat([torch.zeros(T, 1, dtype=torch.long, device=dev),
                        t.unsqueeze(1) - torch.arange(Lw, device=dev).unsqueeze(0)], 1)
        st = st.masked_fill(st < 0, -1).view(1, 1, T, 1 + Lw).expand(b, H, T, 1 + Lw)
        cand = torch.cat([tree_pos, nb_pos, st], -1)

        used = leaf_ok & (torch.arange(wl, device=dev) < sel['n_tree'].unsqueeze(-1))
        tv_used = torch.cat([used, used], -1) & (tree_pos >= 0)
        rest = torch.cat([nb_pos >= 0, st >= 0], -1)
        keep = dedupe(cand, torch.cat([tv_used, rest], -1))

        gidx = cand.clamp(min=0)
        k_g = gather_nodes(k, gidx)
        v_g = gather_nodes(v, gidx)
        S = torch.einsum('bhtd,bhtkd->bhtk', q, k_g) * scale
        w = torch.softmax(S.masked_fill(~keep, float('-inf')).float(), -1).to(v.dtype)
        out = torch.einsum('bhtk,bhtkd->bhtd', w, v_g)

        self.aux['keys_total'] = keep.sum(-1).float().mean().detach()
        self.aux['keys_tree'] = keep[..., :2 * wl].sum(-1).float().mean().detach()
        self.aux['keys_nb'] = keep[..., 2 * wl:2 * wl + NBK].sum(-1).float().mean().detach()
        self.aux['nb_leaves'] = sel['nb_leaves'].float().mean()
        self.aux['leaves_used'] = used.sum(-1).float().mean()

        if self.training and cfg.beam_mode == 'pred':
            with torch.no_grad():
                tv_full = torch.cat([leaf_ok, leaf_ok], -1) & (tree_pos >= 0)
                keep_f = dedupe(cand, torch.cat([tv_full, rest], -1))
                wf = torch.softmax(S.detach().masked_fill(~keep_f, float('-inf')).float(), -1)
                wt = wf[..., :2 * wl]
                leaf_mass = wt[..., :wl] + wt[..., wl:]
                total = leaf_mass.sum(-1, keepdim=True)
                need = (leaf_mass.cumsum(-1) < 0.9 * total).sum(-1) + 1
                label = (need.unsqueeze(-1) > self.classes).sum(-1).clamp(max=len(self.classes) - 1)
                rows = total.squeeze(-1) > 0
            if bool(rows.any()):
                self.aux['pred'] = F.cross_entropy(plog.float()[rows], label[rows])
        return out
