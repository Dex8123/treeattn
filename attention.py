import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .forest import build_forest, gather_nodes


class TreeAttention(nn.Module):
    """Causal block-tree sparse attention.

    Every query attends to
      1. its own block, exactly (causal),
      2. three static keys: the sink (position 0) and the two previous tokens,
      3. the keys it finds by searching the trees of all EARLIER, completed blocks.
    No tree ever contains a future token, so nothing leaks from the future."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C = cfg.n_embd
        self.H = cfg.n_head
        self.dh = C // cfg.n_head
        assert cfg.seq_len % cfg.tree_block == 0
        assert cfg.tree_block >= 4 and (cfg.tree_block & (cfg.tree_block - 1)) == 0
        self.c_attn = nn.Linear(C, 3 * C, bias=False)
        self.c_proj = nn.Linear(C, C, bias=False)
        mask = torch.ones(1, 1, cfg.seq_len, 1)
        mask[:, :, 0, :] = 0          # the sink is left out of the index vectors (it is still always attended)
        self.register_buffer('idx_mask', mask, persistent=False)
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
        B = cfg.tree_block
        assert T % B == 0 and T <= cfg.seq_len
        nb = T // B
        dev = q.device
        scale = 1.0 / math.sqrt(dh)
        t = torch.arange(T, device=dev)
        blk = t // B
        self.aux = {}

        # (1) exact causal attention inside the query's own block
        qb, kb, vb = [x.reshape(b, H, nb, B, dh) for x in (q, k, v)]
        S_loc = torch.matmul(qb, kb.transpose(-1, -2)) * scale
        causal = torch.tril(torch.ones(B, B, dtype=torch.bool, device=dev))
        S_loc = S_loc.masked_fill(~causal, float('-inf')).reshape(b, H, T, B)
        if nb == 1:
            w = torch.softmax(S_loc.float(), -1).to(v.dtype)
            return torch.matmul(w.reshape(b, H, nb, B, B), vb).reshape(b, H, T, dh)

        # (2) one content-sorted tree per block, built from that block's keys only
        depth = int(round(math.log2(B // 2)))
        k_idx = k * self.idx_mask[:, :, :T].to(k.dtype)
        perm_flat, sums = build_forest(k_idx, B, depth, cfg)
        pred_mode = cfg.beam_mode == 'pred'
        bq_desc = torch.full((b, H, T), cfg.wmax if pred_mode else cfg.beam, dtype=torch.long, device=dev)

        # (3) descent: score the roots of all earlier blocks, keep the best, go down level by level
        ar_nb = torch.arange(nb, device=dev)
        cur = ar_nb.view(1, 1, 1, nb).expand(b, H, T, nb)
        cv = (ar_nb[None, :] < blk[:, None]).view(1, 1, T, nb).expand(b, H, T, nb)
        levels = []
        for lvl in range(depth + 1):
            Wc = cur.shape[-1]
            U = gather_nodes(sums[lvl], cur)
            sc = torch.einsum('bhtd,bhtwd->bhtw', q, U).masked_fill(~cv, float('-inf'))
            w_keep = min(cfg.wmax, Wc)
            order = sc.topk(w_keep, dim=-1).indices
            sel = cur.gather(-1, order)
            selv = cv.gather(-1, order) & (torch.arange(w_keep, device=dev) < bq_desc.unsqueeze(-1))
            levels.append((sc, cur, cv))
            if lvl < depth:
                cur = torch.cat([2 * sel, 2 * sel + 1], -1)
                cv = torch.cat([selv, selv], -1)
        leaf, leafv = sel, selv
        wl = leaf.shape[-1]

        # (4) optional learned budget head: how many of the found leaves does this query really use?
        plog = None
        leafv_use = leafv
        if pred_mode:
            qd = q.detach()
            feats = [qd,
                     torch.log(qd.float().norm(dim=-1, keepdim=True) + 1e-6).to(qd.dtype),
                     self.head_eye.to(qd.dtype)[None, :, None, :].expand(b, H, T, H)]
            if cfg.global_pool:
                cum = sums[0].detach().cumsum(2)
                prev = torch.cat([torch.zeros_like(cum[:, :, :1]), cum[:, :, :-1]], 2)
                denom = (ar_nb.clamp(min=1) * B).to(qd.dtype).view(1, 1, nb, 1)
                feats.append((prev / denom)[:, :, blk, :])
            plog = self.pred(torch.cat(feats, -1))
            bq_pred = self.classes[plog.argmax(-1).detach()]
            leafv_use = leafv & (torch.arange(wl, device=dev) < bq_pred.unsqueeze(-1))

        # (5) turn leaves into key positions (2 keys per leaf), drop duplicates of the static keys
        pf = torch.cat([2 * leaf, 2 * leaf + 1], -1)
        loc = perm_flat.gather(2, pf.reshape(b, H, -1)).reshape(b, H, T, -1)
        tkey = (pf // B) * B + loc
        s_idx = torch.stack([torch.zeros_like(t), t - 1, t - 2], -1)
        s_valid = (s_idx >= 0) & (s_idx < (blk * B)[:, None])
        dup = ((tkey.unsqueeze(-1) == s_idx.view(1, 1, T, 1, 3)) & s_valid.view(1, 1, T, 1, 3)).any(-1)
        tvalid_full = torch.cat([leafv, leafv], -1) & ~dup
        tvalid = torch.cat([leafv_use, leafv_use], -1) & ~dup

        # (6) exact attention over: own block + static keys + tree keys
        s_exp = s_idx.view(1, 1, T, 3).expand(b, H, T, 3)
        gidx = torch.cat([s_exp, tkey], -1).clamp(0, T - 1)
        k_g = gather_nodes(k, gidx)
        v_g = gather_nodes(v, gidx)
        Sg = torch.einsum('bhtd,bhtkd->bhtk', q, k_g) * scale
        sv = s_valid.view(1, 1, T, 3).expand(b, H, T, 3)
        gvalid = torch.cat([sv, tvalid], -1)
        allS = torch.cat([S_loc, Sg.masked_fill(~gvalid, float('-inf'))], -1)
        w = torch.softmax(allS.float(), -1).to(v.dtype)
        out = torch.matmul(w[..., :B].reshape(b, H, nb, B, B), vb).reshape(b, H, T, dh) \
            + torch.einsum('bhtk,bhtkd->bhtd', w[..., B:], v_g)

        tcount = tvalid.sum(-1).float()
        self.aux['keys_tree'] = tcount[:, :, B:].mean().detach()
        self.aux['keys_total'] = ((t % B + 1).float().mean() + sv.float().sum(-1).mean() + tcount.mean()).detach()

        # (7) training signals: routing loss (and budget-head loss), labels come from the attention itself
        if self.training:
            with torch.no_grad():
                if pred_mode:
                    gv_full = torch.cat([sv, tvalid_full], -1)
                    w_lab = torch.softmax(torch.cat([S_loc, Sg.masked_fill(~gv_full, float('-inf'))], -1).float(), -1)
                    tv_lab = tvalid_full
                else:
                    w_lab = w.float()
                    tv_lab = tvalid
                wt = w_lab[..., B + 3:]
                best = wt.argmax(-1)
                ok = tv_lab.gather(-1, best.unsqueeze(-1)).squeeze(-1) & (wt.max(-1).values > 0) \
                    & (blk >= 1).view(1, 1, T)
                leafnode = leaf.gather(-1, (best % wl).unsqueeze(-1)).squeeze(-1)
                if pred_mode:
                    leaf_mass = wt[..., :wl] + wt[..., wl:]
                    total = leaf_mass.sum(-1, keepdim=True)
                    need = (leaf_mass.cumsum(-1) < 0.9 * total).sum(-1) + 1
                    label = (need.unsqueeze(-1) > self.classes).sum(-1).clamp(max=len(self.classes) - 1)
            route = q.new_zeros(()).float()
            if cfg.route_weight > 0 and bool(ok.any()):
                terms = []
                for lvl, (sc, cur_l, cv_l) in enumerate(levels):
                    anc = leafnode >> (depth - lvl)
                    match = (cur_l == anc.unsqueeze(-1)) & cv_l
                    rows = ok & match.any(-1)
                    if bool(rows.any()):
                        size = B // (2 ** lvl)
                        lg = (sc / (size * math.sqrt(dh))).float()[rows]
                        terms.append(F.cross_entropy(lg, match.float().argmax(-1)[rows]))
                if terms:
                    route = torch.stack(terms).mean()
            self.aux['route'] = route
            if pred_mode:
                prow = (blk >= 1).view(1, 1, T).expand(b, H, T)
                self.aux['pred'] = F.cross_entropy(plog.float()[prow], label[prow])
        return out
