import torch


def gather_nodes(S, idx):
    """S: (b, H, M, d); idx: (b, H, T, W) long -> (b, H, T, W, d)"""
    b, H, T, W = idx.shape
    d = S.shape[-1]
    out = S.gather(2, idx.reshape(b, H, T * W, 1).expand(-1, -1, -1, d))
    return out.reshape(b, H, T, W, d)


@torch.no_grad()
def two_means_order(X, iters):
    """X: (P, G, S, D) -> order (P, G, S). Inside every group the keys are sorted by a 2-means projection,
    so that cutting the sorted list in half gives two balanced clusters of similar keys."""
    P, G, S, D = X.shape
    mean = X.mean(2, keepdim=True)
    a_idx = ((X - mean) ** 2).sum(-1).argmax(-1)
    a = torch.gather(X, 2, a_idx[..., None, None].expand(-1, -1, 1, D))
    b_idx = ((X - a) ** 2).sum(-1).argmax(-1)
    b = torch.gather(X, 2, b_idx[..., None, None].expand(-1, -1, 1, D))
    c = torch.cat([a, b], 2)
    xx = (X ** 2).sum(-1, keepdim=True)
    for _ in range(iters):
        dist = xx - 2 * torch.einsum('pgsd,pgcd->pgsc', X, c) + (c ** 2).sum(-1)[:, :, None, :]
        lab = dist.argmin(-1)
        for j in (0, 1):
            m = (lab == j).unsqueeze(-1).to(X.dtype)
            cnt = m.sum(2, keepdim=True)
            newc = (X * m).sum(2, keepdim=True) / cnt.clamp(min=1)
            c[:, :, j:j + 1] = torch.where(cnt > 0, newc, c[:, :, j:j + 1])
    proj = torch.einsum('pgsd,pgd->pgs', X, c[:, :, 0] - c[:, :, 1])
    return proj.argsort(-1)


def soft_gate(sm, grp, cfg):
    """Soft version of the mean-log dilution gate.
    sm: (P, n, d) summed keys of every node; grp: (P, n, S, d) the member keys.
    Dimensions whose mean log-magnitude is low inside the group are faded out, the rest are rescaled."""
    ml = torch.log(grp.float().abs() + 1e-6).mean(2)
    thr = torch.quantile(ml, cfg.gate_q, dim=-1, keepdim=True)
    g = torch.sigmoid((ml - thr) / cfg.gate_temp)
    kept = g.sum(-1, keepdim=True).clamp(min=1.0)
    return (sm.float() * g * (sm.shape[-1] / kept)).to(sm.dtype)


def build_forest(k, B, depth, cfg):
    """One content-sorted binary tree per block of B keys (a block's tree only ever sees that block's keys).
    k: (b, H, T, d).
    Returns perm_flat (b, H, nb*B): rank -> local position inside the block, and
    sums[l] (b, H, nb*2**l, d): index vector of every node at level l
    (level 0 = block root, level `depth` = groups of 2 keys). Flat node id = block * 2**l + m."""
    b, H, T, d = k.shape
    nb = T // B
    P = b * H * nb
    kb = k.reshape(P, B, d)
    with torch.no_grad():
        kd = kb.detach().float()
        perm = torch.arange(B, device=k.device).unsqueeze(0).expand(P, B).contiguous()
        for lvl in range(depth):
            G = 2 ** lvl
            S = B // G
            Xg = kd.gather(1, perm.unsqueeze(-1).expand(-1, -1, d)).reshape(P, G, S, d)
            order = two_means_order(Xg, cfg.kmeans_iters)
            perm = perm.reshape(P, G, S).gather(2, order).reshape(P, B)
    kp = kb.gather(1, perm.unsqueeze(-1).expand(-1, -1, d))
    sums = []
    for lvl in range(depth + 1):
        n_nodes = 2 ** lvl
        grp = kp.reshape(P, n_nodes, B // n_nodes, d)
        sm = grp.sum(2)
        if cfg.gate:
            sm = soft_gate(sm, grp, cfg)
        sums.append(sm.reshape(b, H, nb * n_nodes, d))
    return perm.reshape(b, H, nb * B), sums
