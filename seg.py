import math

import torch

from .forest import gather_nodes

# Static positional tree ("segment tree") attention.
# Level l has T >> l nodes, node m covers positions [m * 2**l, (m + 1) * 2**l). A parent is the SUM of its two
# children (additive composition). For a query at position t, the nodes that lie completely in its past are exactly
# the left siblings along the path from the root to t: at every level l where bit l of t is 1, node (t >> l) - 1.
# That is at most log2(T) subtrees and together with position t-1 (when t is odd) they cover [0, t) exactly.
# Nothing here ever touches a node containing t or anything later, so no future token can be selected.
# Leaf groups are the level-1 nodes (2 keys each): leaf m = positions 2m and 2m + 1.


def gate_apply(raw, lgsum, size, gate_q):
    """Mean-log dilution gate: keep the dimensions whose mean log-magnitude inside the group is above the
    gate_q-quantile, zero the rest, rescale the kept ones."""
    ml = lgsum / size
    thr = torch.quantile(ml, gate_q, dim=-1, keepdim=True)
    g = (ml >= thr).to(raw.dtype)
    kept = g.sum(-1, keepdim=True).clamp(min=1.0)
    return raw * g * (raw.shape[-1] / kept)


@torch.no_grad()
def level_sums(kidx, Lm, gate, gate_q):
    """kidx: (b, H, T, dh) index keys (sink zeroed). Returns {l: (b, H, T >> l, dh)} index vectors for l = 1..Lm.
    Parents = sum of children; the mean-log statistic is a sum of logs and composes the same way."""
    b, H, T, dh = kidx.shape
    raw = kidx
    lg = torch.log(kidx.abs() + 1e-6) if gate else None
    out = {}
    for l in range(1, Lm + 1):
        raw = raw.reshape(b, H, T >> l, 2, dh).sum(3)
        if gate:
            lg = lg.reshape(b, H, T >> l, 2, dh).sum(3)
            out[l] = gate_apply(raw, lg, 1 << l, gate_q)
        else:
            out[l] = raw
    return out


@torch.no_grad()
def tree_pass(q, getU, tq, Lm, wmax):
    """Beam descent over the past-only subtrees of every query.
    q: (b, H, Nq, dh); getU(l, cand) -> index vectors of nodes cand (b, H, Nq, C) at level l; tq: (Nq,) positions.
    At every level the candidates are the children of the nodes kept so far plus the newly available past-only
    subtree of that level; the best wmax by score (query . mean key of the node) are kept.
    Returns leaf ids (level 1) (b, H, Nq, wl), validity and scores, best first."""
    b, H, Nq, dh = q.shape
    ids = vld = None
    sct = None
    for l in range(Lm, 0, -1):
        bit = ((tq >> l) & 1) == 1
        inj = ((tq >> l) - 1).clamp(min=0).view(1, 1, Nq, 1).expand(b, H, Nq, 1)
        injv = bit.view(1, 1, Nq, 1).expand(b, H, Nq, 1)
        if ids is None:
            cand, cv = inj, injv
        else:
            cand = torch.cat([2 * ids, 2 * ids + 1, inj], -1)
            cv = torch.cat([vld, vld, injv], -1)
        U = getU(l, cand)
        sc = (q.unsqueeze(3) * U).sum(-1) / float(1 << l)
        sc = sc.masked_fill(~cv, float('-inf'))
        w = min(wmax, cand.shape[-1])
        sct, order = sc.topk(w, dim=-1)
        ids = cand.gather(-1, order)
        vld = cv.gather(-1, order)
    return ids, vld, sct


@torch.no_grad()
def neighbor_pass(q, getU1, st_leaf, st_score, st_ok, tq, cfg):
    """Borrow the best leaf groups that the nearest earlier queries stored, keep those this query also scores well
    on; the scan width is a variable cap set by how well the nearest neighbours match.
    st_*: (b, H, Tcap, NS) stored per position. Returns ids (b, H, Nq, Wn*NS), match mask and the number of
    distinct matched groups nL (b, H, Nq)."""
    b, H, Nq, dh = q.shape
    Wn, NS = cfg.nb_window, cfg.nb_store
    dev = q.device
    back = torch.arange(1, Wn + 1, device=dev).view(1, Wn)
    jpos = tq.view(Nq, 1) - back
    jval = jpos >= 0
    jc = jpos.clamp(min=0)
    ids = st_leaf[:, :, jc].reshape(b, H, Nq, Wn * NS)
    base = st_score[:, :, jc].reshape(b, H, Nq, Wn * NS)
    ok0 = (st_ok[:, :, jc] & jval.view(1, 1, Nq, Wn, 1)).reshape(b, H, Nq, Wn * NS)
    U = getU1(ids)
    sc_t = (q.unsqueeze(3) * U).sum(-1) / 2.0
    match = ok0 & (sc_t >= base - cfg.nb_rho * base.abs())
    probe = cfg.nb_probe * NS
    rate = match[..., :probe].float().sum(-1) / ok0[..., :probe].float().sum(-1).clamp(min=1)
    cap = cfg.nb_cap_min + torch.round((Wn - cfg.nb_cap_min) * rate).long()
    didx = torch.arange(Wn * NS, device=dev) // NS + 1
    match = match & (didx <= cap.unsqueeze(-1))
    if not cfg.neighbors:
        match = torch.zeros_like(match)
    idm = torch.where(match, ids, torch.full_like(ids, -1))
    ss, _ = idm.sort(-1)
    vs = ss >= 0
    nL = vs.sum(-1) - ((ss[..., 1:] == ss[..., :-1]) & vs[..., 1:]).sum(-1)
    return ids, match, nL


def leaves_to_pos(ids, ok):
    pos = torch.cat([2 * ids, 2 * ids + 1], -1)
    return pos.masked_fill(~torch.cat([ok, ok], -1), -1)


def finish(ids, vld, nid, match, nL, lhat, cfg):
    """Turn the budget (lhat leaf groups) into how many tree leaves to use after the neighbour search."""
    L = lhat
    lo = (L + 1) // 2
    need = (L - nL).clamp(min=0)
    floor_b = torch.where(nL < lo, torch.full_like(L, 2), torch.ones_like(L))
    b_tree = torch.where(nL >= L, torch.zeros_like(L), torch.maximum(need, floor_b)).clamp(max=cfg.wmax)
    return dict(tree_pos=leaves_to_pos(ids, vld), leaf_ok=vld, nb_pos=leaves_to_pos(nid, match),
                n_tree=b_tree, nb_leaves=nL)


@torch.no_grad()
def select_seg(q, k, lhat, cfg):
    """Key selection for a whole training sequence at once (fully parallel, no sequential loop).
    Two passes: (1) tree search for every query, (2) neighbour borrowing from earlier queries' pass-1 results."""
    b, H, T, dh = q.shape
    Lm = int(round(math.log2(T))) - 1
    qf = q.float()
    kidx = k.float().clone()
    kidx[:, :, 0, :] = 0
    S = level_sums(kidx, Lm, cfg.gate, cfg.gate_q)

    def getU(l, c):
        return gather_nodes(S[l], c)

    tq = torch.arange(T, device=q.device)
    ids, vld, sct = tree_pass(qf, getU, tq, Lm, cfg.wmax)
    NS = cfg.nb_store
    nid, match, nL = neighbor_pass(qf, lambda c: getU(1, c), ids[..., :NS], sct[..., :NS], vld[..., :NS], tq, cfg)
    return finish(ids, vld, nid, match, nL, lhat, cfg)
