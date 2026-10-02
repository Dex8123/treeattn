import math

import torch


def masked_two_means_order(X, valid, iters):
    """X: (P, G, S, D); valid: (P, G, S) bool. Returns order (P, G, S): real keys first, sorted by a
    2-means projection (similar keys end up next to each other), empty slots last."""
    P, G, S, D = X.shape
    vm = valid.unsqueeze(-1).to(X.dtype)
    cnt = vm.sum(2, keepdim=True).clamp(min=1)
    mean = (X * vm).sum(2, keepdim=True) / cnt
    d2 = ((X - mean) ** 2).sum(-1).masked_fill(~valid, -1.0)
    a_idx = d2.argmax(-1)
    a = torch.gather(X, 2, a_idx[..., None, None].expand(-1, -1, 1, D))
    d2b = ((X - a) ** 2).sum(-1).masked_fill(~valid, -1.0)
    b_idx = d2b.argmax(-1)
    b = torch.gather(X, 2, b_idx[..., None, None].expand(-1, -1, 1, D))
    c = torch.cat([a, b], 2)
    xx = (X ** 2).sum(-1, keepdim=True)
    for _ in range(iters):
        dist = xx - 2 * torch.einsum('pgsd,pgcd->pgsc', X, c) + (c ** 2).sum(-1)[:, :, None, :]
        lab = dist.argmin(-1)
        for j in (0, 1):
            m = ((lab == j) & valid).unsqueeze(-1).to(X.dtype)
            cj = m.sum(2, keepdim=True)
            newc = (X * m).sum(2, keepdim=True) / cj.clamp(min=1)
            c[:, :, j:j + 1] = torch.where(cj > 0, newc, c[:, :, j:j + 1])
    proj = torch.einsum('pgsd,pgd->pgs', X, c[:, :, 0] - c[:, :, 1])
    proj = proj.masked_fill(~valid, float('inf'))
    return proj.argsort(-1)


class LaneTree:
    """A fixed-capacity balanced binary tree over the keys seen so far, one tree per lane.

    Capacity C = number of slots (a power of two). Unused slots are 'empty vectors': they are left out of every
    sum and only keep the tree balanced. Every node holds the SUM of the keys below it (index vector) and a count.
    The tree starts from the keys [1, n) (position 0, the attention sink, is never indexed), split by similarity,
    and then grows by insertion: a new key walks down towards the child whose mean is more similar to it
    (children that are full are skipped) and is added to the sums on its path. Cost per insertion: O(depth).
    Level l has 2**l nodes; the last level (depth D = log2(C) - 1) holds groups of 2 slots."""

    def __init__(self, kfull, n, iters=3):
        P, C, dh = kfull.shape
        self.P, self.C, self.dh = P, C, dh
        self.D = int(round(math.log2(C))) - 1
        self.dev = kfull.device
        self.kfull = kfull
        self.ar = torch.arange(P, device=self.dev)
        D = self.D
        perm = torch.arange(C, device=self.dev).unsqueeze(0).expand(P, C).contiguous()
        valid = (perm >= 1) & (perm < n.unsqueeze(1))
        for lvl in range(D):
            G = 2 ** lvl
            S = C // G
            Xg = kfull.gather(1, perm.unsqueeze(-1).expand(-1, -1, dh)).reshape(P, G, S, dh)
            vg = valid.reshape(P, G, S)
            pg = perm.reshape(P, G, S)
            order = masked_two_means_order(Xg, vg, iters)
            pg = pg.gather(2, order)
            vg = vg.gather(2, order)
            ng = vg.sum(-1)
            nl = (ng + 1) // 2
            nr = ng - nl
            r = torch.arange(S, device=self.dev).view(1, 1, S)
            half = S // 2
            left = r < half
            src_real = torch.where(left, r, nl.unsqueeze(-1) + (r - half))
            cond = torch.where(left, r < nl.unsqueeze(-1), (r - half) < nr.unsqueeze(-1))
            src = torch.where(cond, src_real, torch.full_like(src_real, S - 1))
            pg = pg.gather(2, src)
            perm = pg.reshape(P, C)
            valid = cond.reshape(P, C)
        kp = kfull.gather(1, perm.unsqueeze(-1).expand(-1, -1, dh)) * valid.unsqueeze(-1).to(kfull.dtype)
        self.sums, self.cnt = [], []
        for lvl in range(D + 1):
            nn = 2 ** lvl
            self.sums.append(kp.reshape(P, nn, C // nn, dh).sum(2).contiguous())
            self.cnt.append(valid.reshape(P, nn, C // nn).sum(2).contiguous())
        self.slot = torch.where(valid, perm, torch.full_like(perm, -1)).reshape(P, 2 ** D, 2).contiguous()

    def insert(self, kv, kpos, active):
        """kv: (P, dh) new key vectors; kpos: (P,) their positions; active: (P,) bool (False = do nothing)."""
        ar = self.ar
        act = active.long()
        kva = kv * act.unsqueeze(-1).to(kv.dtype)
        node = torch.zeros(self.P, dtype=torch.long, device=self.dev)
        self.sums[0].index_put_((ar, node), kva, accumulate=True)
        self.cnt[0].index_put_((ar, node), act, accumulate=True)
        for lvl in range(self.D):
            left = 2 * node
            right = left + 1
            cap = self.C >> (lvl + 1)
            cL = self.cnt[lvl + 1][ar, left]
            cR = self.cnt[lvl + 1][ar, right]
            sL = (kv * self.sums[lvl + 1][ar, left]).sum(-1) / cL.clamp(min=1)
            sR = (kv * self.sums[lvl + 1][ar, right]).sum(-1) / cR.clamp(min=1)
            sL = torch.where(cL == 0, torch.full_like(sL, -1e9), sL).masked_fill(cL >= cap, float('-inf'))
            sR = torch.where(cR == 0, torch.full_like(sR, -1e9), sR).masked_fill(cR >= cap, float('-inf'))
            node = torch.where(sR > sL, right, left)
            self.sums[lvl + 1].index_put_((ar, node), kva, accumulate=True)
            self.cnt[lvl + 1].index_put_((ar, node), act, accumulate=True)
        slot = (self.slot[ar, node, 0] >= 0).long()
        old = self.slot[ar, node, slot]
        self.slot[ar, node, slot] = torch.where(active, kpos, old)

    def search(self, qt, wmax):
        """Beam descent for one query per lane. Returns leaf ids (P, wl), validity (P, wl), scores (P, wl),
        sorted best first. A node is scored by the dot product of the query with its MEAN key."""
        P, dh = self.P, self.dh
        cur = torch.zeros(P, 1, dtype=torch.long, device=self.dev)
        cvld = self.cnt[0] > 0
        sc_top = None
        for lvl in range(1, self.D + 1):
            ch = torch.cat([2 * cur, 2 * cur + 1], -1)
            cv = torch.cat([cvld, cvld], -1)
            cch = self.cnt[lvl].gather(1, ch)
            cv = cv & (cch > 0)
            sm = self.sums[lvl].gather(1, ch.unsqueeze(-1).expand(-1, -1, dh))
            sc = (qt.unsqueeze(1) * sm).sum(-1) / cch.clamp(min=1)
            sc = sc.masked_fill(~cv, float('-inf'))
            w = min(wmax, ch.shape[-1])
            sc_top, order = sc.topk(w, dim=-1)
            cur = ch.gather(1, order)
            cvld = cv.gather(1, order)
        return cur, cvld, sc_top

    def leaf_keys(self, ids):
        """ids: (P, W) leaf node ids -> key positions (P, W, 2), -1 where a slot is empty."""
        return self.slot.gather(1, ids.unsqueeze(-1).expand(-1, -1, 2))

    def check(self):
        """Recompute every node sum and count from the leaf slots; returns the worst mismatch (0 = consistent)."""
        P, C, dh = self.P, self.C, self.dh
        pos = self.slot.reshape(P, C)
        valid = pos >= 0
        kp = self.kfull.gather(1, pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, dh)) \
            * valid.unsqueeze(-1).to(self.kfull.dtype)
        worst = 0.0
        for lvl in range(self.D + 1):
            nn = 2 ** lvl
            s = kp.reshape(P, nn, C // nn, dh).sum(2)
            c = valid.reshape(P, nn, C // nn).sum(2)
            worst = max(worst, (s - self.sums[lvl]).abs().max().item(), float((c != self.cnt[lvl]).sum().item()))
            assert (self.cnt[lvl] <= C // nn).all(), 'node over capacity at level %d' % lvl
        return worst


@torch.no_grad()
def select_keys(q, k, lhat, cfg):
    """Non-differentiable key selection for every query, lane by lane.

    The sequence is cut into lanes of cfg.chunk tokens that are processed in parallel (one extra tensor dimension).
    Lane l starts with a tree built from all keys before its first token, then walks through its tokens one by one:
      1. neighbour search: borrow the best leaf groups that the nearest earlier queries of the same lane found,
         keep the ones this query also scores well on; the scan width is a variable cap set by how well the
         nearest neighbours match,
      2. tree search for the rest of the budget lhat (leaves): beam descent over the current tree,
      3. insert this token's key into the tree.
    Only keys before the query are ever in the tree, so nothing from the future can be selected.
    q, k: (b, H, T, dh); lhat: (b, H, T) leaf budget per query."""
    b, H, T, dh = q.shape
    Y = cfg.chunk
    assert T % Y == 0 and (T & (T - 1)) == 0
    nl = T // Y
    P = b * H * nl
    dev = q.device
    wmax, Wn, NS = cfg.wmax, cfg.nb_window, cfg.nb_store
    D = int(round(math.log2(T))) - 1
    wl = min(wmax, 2 ** D)
    qf, kf = q.float(), k.float()
    starts = torch.arange(nl, device=dev) * Y
    kfull = kf.unsqueeze(2).expand(b, H, nl, T, dh).reshape(P, T, dh)
    n0 = starts.view(1, 1, nl).expand(b, H, nl).reshape(P)
    tree = LaneTree(kfull, n0, cfg.kmeans_iters)

    tree_pos = torch.full((b, H, T, 2 * wl), -1, dtype=torch.long, device=dev)
    leaf_ok = torch.zeros(b, H, T, wl, dtype=torch.bool, device=dev)
    nb_pos = torch.full((b, H, T, Wn * NS * 2), -1, dtype=torch.long, device=dev)
    n_tree = torch.zeros(b, H, T, dtype=torch.long, device=dev)
    nb_leaves = torch.zeros(b, H, T, dtype=torch.long, device=dev)
    st_leaf = torch.zeros(b, H, T, NS, dtype=torch.long, device=dev)
    st_score = torch.zeros(b, H, T, NS, device=dev)
    st_ok = torch.zeros(b, H, T, NS, dtype=torch.bool, device=dev)
    dist_idx = (torch.arange(Wn * NS, device=dev) // NS + 1).unsqueeze(0)
    back = torch.arange(1, Wn + 1, device=dev).unsqueeze(0)
    probe = cfg.nb_probe * NS

    for i in range(Y):
        tpos = starts + i
        qt = qf[:, :, tpos, :].reshape(P, dh)
        kt = kf[:, :, tpos, :].reshape(P, dh)
        tposP = tpos.view(1, 1, nl).expand(b, H, nl).reshape(P)
        L = lhat[:, :, tpos].reshape(P)

        # ---- 1. neighbour search (only earlier queries of the same lane)
        jpos = tpos.unsqueeze(1) - back
        jval = jpos >= starts.unsqueeze(1)
        jc = jpos.clamp(min=0)
        ids = st_leaf[:, :, jc].reshape(P, Wn * NS)
        base = st_score[:, :, jc].reshape(P, Wn * NS)
        ok0 = (st_ok[:, :, jc] & jval.view(1, 1, nl, Wn, 1)).reshape(P, Wn * NS)
        sm = tree.sums[D].gather(1, ids.unsqueeze(-1).expand(-1, -1, dh))
        cn = tree.cnt[D].gather(1, ids)
        sc_t = (qt.unsqueeze(1) * sm).sum(-1) / cn.clamp(min=1)
        match = ok0 & (cn > 0) & (sc_t >= base - cfg.nb_rho * base.abs())
        rate = match[:, :probe].float().sum(1) / ok0[:, :probe].float().sum(1).clamp(min=1)
        cap = cfg.nb_cap_min + torch.round((Wn - cfg.nb_cap_min) * rate).long()
        match = match & (dist_idx <= cap.unsqueeze(1))
        if not cfg.neighbors:
            match = torch.zeros_like(match)
        idm = torch.where(match, ids, torch.full_like(ids, -1))
        ss, _ = idm.sort(-1)
        vs = ss >= 0
        nL = vs.sum(1) - ((ss[:, 1:] == ss[:, :-1]) & vs[:, 1:]).sum(1)

        # ---- 2. tree search for the rest of the budget
        lo = (L + 1) // 2
        need = (L - nL).clamp(min=0)
        floor_b = torch.where(nL < lo, torch.full_like(L, 2), torch.ones_like(L))
        b_tree = torch.where(nL >= L, torch.zeros_like(L), torch.maximum(need, floor_b)).clamp(max=wmax)
        cur, cvld, sct = tree.search(qt, wmax)
        pos = tree.leaf_keys(cur)
        pos = pos.masked_fill(~((pos >= 0) & cvld.unsqueeze(-1)), -1)
        tp = torch.cat([pos[..., 0], pos[..., 1]], -1)
        npos = tree.leaf_keys(ids)
        npos = npos.masked_fill(~((npos >= 0) & match.unsqueeze(-1)), -1)
        npos = torch.cat([npos[..., 0], npos[..., 1]], -1)

        tree_pos[:, :, tpos] = tp.reshape(b, H, nl, 2 * wl)
        leaf_ok[:, :, tpos] = cvld.reshape(b, H, nl, wl)
        nb_pos[:, :, tpos] = npos.reshape(b, H, nl, Wn * NS * 2)
        n_tree[:, :, tpos] = b_tree.reshape(b, H, nl)
        nb_leaves[:, :, tpos] = nL.reshape(b, H, nl)
        st_leaf[:, :, tpos] = cur[:, :NS].reshape(b, H, nl, NS)
        st_score[:, :, tpos] = sct[:, :NS].reshape(b, H, nl, NS)
        st_ok[:, :, tpos] = cvld[:, :NS].reshape(b, H, nl, NS)

        # ---- 3. this token's key joins the tree (the sink, position 0, is never indexed)
        tree.insert(kt, tposP, tposP >= 1)

    return dict(tree_pos=tree_pos, leaf_ok=leaf_ok, nb_pos=nb_pos, n_tree=n_tree, nb_leaves=nb_leaves)
