"""Fast version of the seg attention (same math as attention_dyn.DynAttention with attn='seg').

What is faster, and why (hypotheses until timed on your GPU, see tests/test_fast.py):
  1. gather_dot: scores of a query against gathered tree nodes in ONE kernel instead of gather + multiply + sum
     (the old path wrote and re-read a (b, H, T, 17, 64) float tensor at each of the ~11 tree levels).
  2. sparse_attn: attention over the ~36 selected keys per query as one fused forward and backward kernel
     (the old path gathered K and V into big tensors and scattered the gradients back).
  3. dedupe_fast: duplicate removal by sorting 36 numbers per query instead of a 36 x 36 comparison table.
  4. the mean-log gate is applied only at tree levels >= cfg.gate_min_level.
Every kernel has a plain PyTorch fallback, and the kernels are checked against it once at start-up; if the check
fails (or Triton / a supported GPU is missing) the fallback is used and a message is printed. Nothing here is
quadratic: all work per query is bounded by the settings.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .forest import gather_nodes
from .seg import finish, gate_apply

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # no Triton: fallbacks are used
    HAS_TRITON = False

FAST = [True]  # set to False (long_train.py --fast 0) to force the PyTorch fallbacks
_STATE = {'checked': False, 'ok': False}

if HAS_TRITON:
    @triton.jit
    def _gather_dot_kernel(Q, S, CAND, OUT, N, M, scale, C: tl.constexpr, DH: tl.constexpr, BQ: tl.constexpr):
        pid = tl.program_id(0)
        bh = tl.program_id(1)
        offs_q = pid * BQ + tl.arange(0, BQ)
        mq = offs_q < N
        offs_d = tl.arange(0, DH)
        row = bh * N + offs_q
        q = tl.load(Q + row[:, None] * DH + offs_d[None, :], mask=mq[:, None], other=0.0).to(tl.float32)
        for c in tl.static_range(C):
            idx = tl.load(CAND + row * C + c, mask=mq, other=0)
            srow = tl.load(S + (bh * M + idx)[:, None] * DH + offs_d[None, :], mask=mq[:, None], other=0.0).to(tl.float32)
            val = tl.sum(q * srow, axis=1) * scale
            tl.store(OUT + row * C + c, val, mask=mq)

    @triton.jit
    def _attn_fwd(Q, K, V, CAND, O, LSE, T, scale, K_: tl.constexpr, DH: tl.constexpr, BQ: tl.constexpr):
        pid = tl.program_id(0)
        bh = tl.program_id(1)
        offs_q = pid * BQ + tl.arange(0, BQ)
        mq = offs_q < T
        offs_d = tl.arange(0, DH)
        row = bh * T + offs_q
        q = tl.load(Q + row[:, None] * DH + offs_d[None, :], mask=mq[:, None], other=0.0).to(tl.float32)
        m = tl.full([BQ], -1e30, dtype=tl.float32)
        l = tl.zeros([BQ], dtype=tl.float32)
        acc = tl.zeros([BQ, DH], dtype=tl.float32)
        for j in tl.static_range(K_):
            idx = tl.load(CAND + row * K_ + j, mask=mq, other=-1)
            valid = (idx >= 0) & mq
            kr = bh * T + tl.where(valid, idx, 0)
            kk = tl.load(K + kr[:, None] * DH + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            s = tl.sum(q * kk, axis=1) * scale
            s = tl.where(valid, s, -1e30)
            m_new = tl.maximum(m, s)
            alpha = tl.exp(m - m_new)
            p = tl.where(valid, tl.exp(s - m_new), 0.0)
            vv = tl.load(V + kr[:, None] * DH + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha[:, None] + p[:, None] * vv
            l = l * alpha + p
            m = m_new
        l_safe = tl.where(l > 0, l, 1.0)
        out = acc / l_safe[:, None]
        tl.store(O + row[:, None] * DH + offs_d[None, :], out.to(O.dtype.element_ty), mask=mq[:, None])
        tl.store(LSE + row, m + tl.log(l_safe), mask=mq)

    @triton.jit
    def _attn_bwd(Q, K, V, CAND, O, DO, LSE, DQ, DK, DV, T, scale, K_: tl.constexpr, DH: tl.constexpr,
                  BQ: tl.constexpr):
        pid = tl.program_id(0)
        bh = tl.program_id(1)
        offs_q = pid * BQ + tl.arange(0, BQ)
        mq = offs_q < T
        offs_d = tl.arange(0, DH)
        row = bh * T + offs_q
        io = row[:, None] * DH + offs_d[None, :]
        q = tl.load(Q + io, mask=mq[:, None], other=0.0).to(tl.float32)
        o = tl.load(O + io, mask=mq[:, None], other=0.0).to(tl.float32)
        do = tl.load(DO + io, mask=mq[:, None], other=0.0).to(tl.float32)
        lse = tl.load(LSE + row, mask=mq, other=0.0)
        di = tl.sum(o * do, axis=1)
        dq = tl.zeros([BQ, DH], dtype=tl.float32)
        for j in tl.static_range(K_):
            idx = tl.load(CAND + row * K_ + j, mask=mq, other=-1)
            valid = (idx >= 0) & mq
            kr = bh * T + tl.where(valid, idx, 0)
            ptr = kr[:, None] * DH + offs_d[None, :]
            kk = tl.load(K + ptr, mask=valid[:, None], other=0.0).to(tl.float32)
            vv = tl.load(V + ptr, mask=valid[:, None], other=0.0).to(tl.float32)
            s = tl.sum(q * kk, axis=1) * scale
            p = tl.where(valid, tl.exp(s - lse), 0.0)
            dp = tl.sum(do * vv, axis=1)
            ds = p * (dp - di)
            dq += ds[:, None] * kk * scale
            tl.atomic_add(DK + ptr, ds[:, None] * q * scale, mask=valid[:, None])
            tl.atomic_add(DV + ptr, p[:, None] * do, mask=valid[:, None])
        tl.store(DQ + io, dq.to(DQ.dtype.element_ty), mask=mq[:, None])


def _use_fast(t):
    return FAST[0] and HAS_TRITON and t.is_cuda and fast_available()


# ------------------------------------------------------------------ scores of queries against gathered nodes
def gather_dot(q, S, cand, scale):
    """q (b,H,N,dh), S (b,H,M,dh), cand (b,H,N,C) node ids -> (b,H,N,C) float32 = scale * q . S[cand]."""
    if not _use_fast(q):
        U = gather_nodes(S, cand.clamp(0, S.shape[2] - 1))
        return (q.unsqueeze(3).float() * U.float()).sum(-1) * scale
    b, H, N, dh = q.shape
    M, C = S.shape[2], cand.shape[-1]
    qc, Sc = q.contiguous(), S.contiguous()
    cc = cand.clamp(0, M - 1).to(torch.int32).contiguous()
    out = torch.empty(b, H, N, C, device=q.device, dtype=torch.float32)
    BQ = 32
    _gather_dot_kernel[(triton.cdiv(N, BQ), b * H)](qc, Sc, cc, out, N, M, float(scale), C=C, DH=dh, BQ=BQ, num_warps=4)
    return out


# ------------------------------------------------------------------ attention over the selected keys
def sparse_attn_ref(q, k, v, cand, scale):
    """Reference: cand (b,H,T,K) key positions, -1 = unused. Exact softmax attention over the used keys."""
    keep = cand >= 0
    gidx = cand.clamp(min=0).long()
    k_g, v_g = gather_nodes(k, gidx), gather_nodes(v, gidx)
    S = torch.einsum('bhtd,bhtkd->bhtk', q, k_g) * scale
    w = torch.softmax(S.masked_fill(~keep, float('-inf')).float(), -1).to(v.dtype)
    return torch.einsum('bhtk,bhtkd->bhtd', w, v_g)


class _SparseAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cand, scale):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        b, H, T, dh = q.shape
        K_ = cand.shape[-1]
        o = torch.empty_like(v)
        lse = torch.empty(b, H, T, device=q.device, dtype=torch.float32)
        BQ = 32
        _attn_fwd[(triton.cdiv(T, BQ), b * H)](q, k, v, cand, o, lse, T, float(scale), K_=K_, DH=dh, BQ=BQ, num_warps=4)
        ctx.save_for_backward(q, k, v, cand, o, lse)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, cand, o, lse = ctx.saved_tensors
        do = do.contiguous()
        b, H, T, dh = q.shape
        K_ = cand.shape[-1]
        dq = torch.empty_like(q)
        dk = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dv = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        BQ = 32
        _attn_bwd[(triton.cdiv(T, BQ), b * H)](q, k, v, cand, o, do, lse, dq, dk, dv, T, float(ctx.scale),
                                               K_=K_, DH=dh, BQ=BQ, num_warps=4)
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None


def sparse_attn(q, k, v, cand, scale):
    """cand: (b,H,T,K) int, -1 = unused slot. Differentiable in q, k, v."""
    if _use_fast(q):
        return _SparseAttnFn.apply(q, k, v, cand.to(torch.int32).contiguous(), scale)
    return sparse_attn_ref(q, k, v, cand, scale)


def fast_available():
    """Run the kernels once on small random data and compare with the PyTorch fallback."""
    if not HAS_TRITON or not torch.cuda.is_available():
        return False
    if _STATE['checked']:
        return _STATE['ok']
    _STATE['checked'] = True
    try:
        g = torch.Generator(device='cuda').manual_seed(0)
        b, H, T, dh, K_ = 1, 2, 160, 64, 12
        q, k, v = [torch.randn(b, H, T, dh, device='cuda', generator=g, requires_grad=True) for _ in range(3)]
        cand = torch.randint(-1, T, (b, H, T, K_), device='cuda', generator=g)
        cand[..., 0] = torch.arange(T, device='cuda')
        w = torch.randn(b, H, T, dh, device='cuda', generator=g)
        o1 = _SparseAttnFn.apply(q, k, v, cand.to(torch.int32).contiguous(), 0.125)
        (o1 * w).sum().backward()
        g1 = [t.grad.clone() for t in (q, k, v)]
        for t in (q, k, v):
            t.grad = None
        o2 = sparse_attn_ref(q, k, v, cand, 0.125)
        (o2 * w).sum().backward()
        g2 = [t.grad for t in (q, k, v)]
        errs = [(o1 - o2).abs().max().item()] + [(a - c).abs().max().item() for a, c in zip(g1, g2)]
        S = torch.randn(b, H, 100, dh, device='cuda', generator=g)
        cd = torch.randint(0, 100, (b, H, T, 9), device='cuda', generator=g)
        d1 = (q.detach().float().unsqueeze(3) * gather_nodes(S, cd)).sum(-1) * 0.5
        _STATE['ok'] = True  # needed so gather_dot below takes the kernel path
        d2 = gather_dot(q.detach(), S, cd, 0.5)
        errs.append((d1 - d2).abs().max().item())
        ok = max(errs) < 2e-3
        _STATE['ok'] = ok
        print('fastseg: kernel check %s (max errors out/dq/dk/dv/dot: %s)' % (
            'PASSED' if ok else 'FAILED, using PyTorch fallback', ' '.join('%.1e' % e for e in errs)), flush=True)
    except Exception as e:  # compile or launch problem
        _STATE['ok'] = False
        print('fastseg: kernels unavailable (%s: %s), using PyTorch fallback' % (type(e).__name__, str(e)[:200]), flush=True)
    return _STATE['ok']


# ------------------------------------------------------------------ duplicate removal without a K x K table
def dedupe_fast(cand, valid):
    """Keep the first valid copy of every key position (same result as attention_dyn.dedupe)."""
    key = torch.where(valid, cand, torch.full_like(cand, -1))
    sk, order = torch.sort(key, dim=-1, stable=True)
    dup_sorted = torch.zeros_like(valid)
    dup_sorted[..., 1:] = (sk[..., 1:] == sk[..., :-1]) & (sk[..., 1:] >= 0)
    dup = torch.zeros_like(valid).scatter(-1, order, dup_sorted)
    return valid & ~dup


# ------------------------------------------------------------------ key selection (same logic as seg.select_seg)
@torch.no_grad()
def level_sums_min(kidx, Lm, gate, gate_q, gate_min):
    b, H, T, dh = kidx.shape
    raw = kidx
    lg = torch.log(kidx.abs() + 1e-6) if gate else None
    out = {}
    for l in range(1, Lm + 1):
        raw = raw.reshape(b, H, T >> l, 2, dh).sum(3)
        if gate:
            lg = lg.reshape(b, H, T >> l, 2, dh).sum(3)
            out[l] = gate_apply(raw, lg, 1 << l, gate_q) if l >= gate_min else raw
        else:
            out[l] = raw
    return out


@torch.no_grad()
def tree_pass_fast(q, S, tq, Lm, wmax):
    b, H, Nq, dh = q.shape
    ids = vld = None
    for l in range(Lm, 0, -1):
        bit = ((tq >> l) & 1) == 1
        inj = ((tq >> l) - 1).clamp(min=0).view(1, 1, Nq, 1).expand(b, H, Nq, 1)
        injv = bit.view(1, 1, Nq, 1).expand(b, H, Nq, 1)
        if ids is None:
            cand, cv = inj, injv
        else:
            cand = torch.cat([2 * ids, 2 * ids + 1, inj], -1)
            cv = torch.cat([vld, vld, injv], -1)
        sc = gather_dot(q, S[l], cand, 1.0 / float(1 << l)).masked_fill(~cv, float('-inf'))
        sct, order = sc.topk(min(wmax, cand.shape[-1]), dim=-1)
        ids = cand.gather(-1, order)
        vld = cv.gather(-1, order)
    return ids, vld, sct


@torch.no_grad()
def neighbor_pass_fast(q, S1, st_leaf, st_score, st_ok, tq, cfg):
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
    sc_t = gather_dot(q, S1, ids, 0.5)
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


@torch.no_grad()
def select_seg_fast(q, k, lhat, cfg):
    b, H, T, dh = q.shape
    Lm = int(round(math.log2(T))) - 1
    qf = q.float()
    kidx = k.float().clone()
    kidx[:, :, 0, :] = 0
    S = level_sums_min(kidx, Lm, cfg.gate, cfg.gate_q, getattr(cfg, 'gate_min_level', 0))
    tq = torch.arange(T, device=q.device)
    ids, vld, sct = tree_pass_fast(qf, S, tq, Lm, cfg.wmax)
    NS = cfg.nb_store
    nid, match, nL = neighbor_pass_fast(qf, S[1], ids[..., :NS], sct[..., :NS], vld[..., :NS], tq, cfg)
    return finish(ids, vld, nid, match, nL, lhat, cfg)


# ------------------------------------------------------------------ the module
class SegAttention(nn.Module):
    """Same parameters and same math as DynAttention(attn='seg'); checkpoints load into either."""

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

        sel = select_seg_fast(q.detach(), k.detach(), lhat, cfg)
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
        keep = dedupe_fast(cand, torch.cat([tv_used, rest], -1))

        out = sparse_attn(q, k, v, torch.where(keep, cand, torch.full_like(cand, -1)), scale)

        self.aux['keys_total'] = keep.sum(-1).float().mean().detach()
        self.aux['keys_tree'] = keep[..., :2 * wl].sum(-1).float().mean().detach()
        self.aux['keys_nb'] = keep[..., 2 * wl:2 * wl + NBK].sum(-1).float().mean().detach()
        self.aux['nb_leaves'] = sel['nb_leaves'].float().mean()
        self.aux['leaves_used'] = used.sum(-1).float().mean()

        if self.training and cfg.beam_mode == 'pred':
            with torch.no_grad():
                tv_full = torch.cat([leaf_ok, leaf_ok], -1) & (tree_pos >= 0)
                keep_f = dedupe_fast(cand, torch.cat([tv_full, rest], -1))
                Sf = gather_dot(q.detach(), k.detach(), cand, scale)
                wf = torch.softmax(Sf.masked_fill(~keep_f, float('-inf')), -1)
                wt = wf[..., :2 * wl]
                leaf_mass = wt[..., :wl] + wt[..., wl:]
                total = leaf_mass.sum(-1, keepdim=True)
                need = (leaf_mass.cumsum(-1) < 0.9 * total).sum(-1) + 1
                label = (need.unsqueeze(-1) > self.classes).sum(-1).clamp(max=len(self.classes) - 1)
                rows = total.squeeze(-1) > 0
            if bool(rows.any()):
                self.aux['pred'] = F.cross_entropy(plog.float()[rows], label[rows])
        return out
