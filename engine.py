import math

import torch
import torch.nn.functional as F

from .attention_dyn import dedupe
from .forest import gather_nodes
from .seg import finish, gate_apply, neighbor_pass, tree_pass

# Token-by-token decoding with a key cache, for our segment-tree attention ('seg') and for ordinary dense attention
# ('dense', used for the nanoGPT baseline). Batch size 1, float32.


def _ln(x, w):
    return F.layer_norm(x, (x.shape[-1],), w, None, 1e-5)


class SegState:
    """Per-layer cache: keys, values, the tree of index vectors (a node is formed the moment both of its children
    are complete, by adding them), and the leaf groups each earlier query found (for neighbour borrowing)."""

    def __init__(self, cfg, Tcap, H, dh, dev):
        self.cfg, self.Tcap, self.H, self.dh = cfg, Tcap, H, dh
        self.Lm = int(round(math.log2(Tcap))) - 1
        z = lambda *s: torch.zeros(*s, device=dev)
        self.K, self.V = z(1, H, Tcap, dh), z(1, H, Tcap, dh)
        self.raw = [z(1, H, Tcap >> l, dh) for l in range(self.Lm + 1)]
        self.lg = [z(1, H, Tcap >> l, dh) for l in range(self.Lm + 1)] if cfg.gate else None
        NS = cfg.nb_store
        self.st_leaf = torch.zeros(1, H, Tcap, NS, dtype=torch.long, device=dev)
        self.st_score = z(1, H, Tcap, NS)
        self.st_ok = torch.zeros(1, H, Tcap, NS, dtype=torch.bool, device=dev)
        self.ksum = z(1, H, dh)

    def getU(self, l, cand):
        U = gather_nodes(self.raw[l], cand)
        if self.cfg.gate:
            U = gate_apply(U, gather_nodes(self.lg[l], cand), 1 << l, self.cfg.gate_q)
        return U

    def add(self, t, k, v):
        """k, v: (1, H, dh) for position t."""
        self.K[:, :, t], self.V[:, :, t] = k, v
        kidx = torch.zeros_like(k) if t == 0 else k
        self.raw[0][:, :, t] = kidx
        if self.lg is not None:
            self.lg[0][:, :, t] = torch.log(kidx.abs() + 1e-6)
        for l in range(1, self.Lm + 1):
            if (t + 1) % (1 << l) != 0:
                break
            n = t >> l
            self.raw[l][:, :, n] = self.raw[l - 1][:, :, 2 * n] + self.raw[l - 1][:, :, 2 * n + 1]
            if self.lg is not None:
                self.lg[l][:, :, n] = self.lg[l - 1][:, :, 2 * n] + self.lg[l - 1][:, :, 2 * n + 1]
        self.ksum = self.ksum + k

    def fill(self, K, V):
        """Load a whole context at once (used by the speed benchmark)."""
        H, dh = self.H, self.dh
        self.K, self.V = K.clone(), V.clone()
        kidx = K.clone()
        kidx[:, :, 0] = 0
        self.raw[0] = kidx
        lg = torch.log(kidx.abs() + 1e-6) if self.lg is not None else None
        if lg is not None:
            self.lg[0] = lg
        for l in range(1, self.Lm + 1):
            self.raw[l] = self.raw[l - 1].reshape(1, H, self.Tcap >> l, 2, dh).sum(3)
            if lg is not None:
                self.lg[l] = self.lg[l - 1].reshape(1, H, self.Tcap >> l, 2, dh).sum(3)
        self.ksum = K.sum(2)


class DenseState:
    def __init__(self, Tcap, H, dh, dev):
        self.K = torch.zeros(1, H, Tcap, dh, device=dev)
        self.V = torch.zeros(1, H, Tcap, dh, device=dev)

    def attend(self, q, k, v, t):
        self.K[:, :, t], self.V[:, :, t] = k[:, :, 0], v[:, :, 0]
        return F.scaled_dot_product_attention(q, self.K[:, :, :t + 1], self.V[:, :, :t + 1]), float(t + 1)


@torch.no_grad()
def seg_attend(state, pw, q, k, v, t, update=True):
    """One decoding step of segment-tree attention. q, k, v: (1, H, 1, dh). Returns (out, keys read per head)."""
    cfg, H, dh = state.cfg, state.H, state.dh
    dev = q.device
    if cfg.beam_mode == 'pred':
        feats = [q, torch.log(q.norm(dim=-1, keepdim=True) + 1e-6), torch.eye(H, device=dev).view(1, H, 1, H)]
        if cfg.global_pool:
            feats.append((state.ksum / max(t, 1)).unsqueeze(2))
        h = F.relu(F.linear(torch.cat(feats, -1), pw['w1'], pw['b1']))
        lhat = pw['classes'][F.linear(h, pw['w2'], pw['b2']).argmax(-1)]
    else:
        lhat = torch.full((1, H, 1), cfg.beam, dtype=torch.long, device=dev)
    tq = torch.tensor([t], device=dev)
    ids, vld, sct = tree_pass(q, state.getU, tq, state.Lm, cfg.wmax)
    NS = cfg.nb_store
    nid, match, nL = neighbor_pass(q, lambda c: state.getU(1, c), state.st_leaf, state.st_score, state.st_ok, tq, cfg)
    sel = finish(ids, vld, nid, match, nL, lhat, cfg)
    if update:
        state.st_leaf[:, :, t] = ids[:, :, 0, :NS]
        state.st_score[:, :, t] = sct[:, :, 0, :NS]
        state.st_ok[:, :, t] = vld[:, :, 0, :NS]
        state.add(t, k[:, :, 0], v[:, :, 0])
    wl = ids.shape[-1]
    Lw = max(cfg.local, 3)
    st = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), t - torch.arange(Lw, device=dev)])
    st = st.masked_fill(st < 0, -1).view(1, 1, 1, 1 + Lw).expand(1, H, 1, 1 + Lw)
    cand = torch.cat([sel['tree_pos'], sel['nb_pos'], st], -1)
    used = sel['leaf_ok'] & (torch.arange(wl, device=dev) < sel['n_tree'].unsqueeze(-1))
    tv = torch.cat([used, used], -1) & (sel['tree_pos'] >= 0)
    keep = dedupe(cand, torch.cat([tv, sel['nb_pos'] >= 0, st >= 0], -1))
    gidx = cand.clamp(min=0)
    kg, vg = gather_nodes(state.K, gidx), gather_nodes(state.V, gidx)
    S = torch.einsum('bhtd,bhtkd->bhtk', q, kg) * (1.0 / math.sqrt(dh))
    w = torch.softmax(S.masked_fill(~keep, float('-inf')).float(), -1)
    out = torch.einsum('bhtk,bhtkd->bhtd', w, vg)
    return out, keep.float().sum(-1).mean().item()


def weights_from_ours(sd, cfg):
    W = dict(wte=sd['wte.weight'], wpe=sd['wpe.weight'], ln_f=sd['ln_f.weight'], layers=[])
    for i in range(cfg.n_layer):
        p = 'blocks.%d.' % i
        L = dict(ln1=sd[p + 'ln1.weight'], c_attn=sd[p + 'attn.c_attn.weight'], c_proj=sd[p + 'attn.c_proj.weight'],
                 ln2=sd[p + 'ln2.weight'], c_fc=sd[p + 'mlp.c_fc.weight'], c_mlp=sd[p + 'mlp.c_proj.weight'])
        if cfg.beam_mode == 'pred':
            L['pw'] = dict(w1=sd[p + 'attn.pred.0.weight'], b1=sd[p + 'attn.pred.0.bias'],
                           w2=sd[p + 'attn.pred.2.weight'], b2=sd[p + 'attn.pred.2.bias'],
                           classes=torch.tensor(list(cfg.beam_classes), device=sd['wte.weight'].device))
        W['layers'].append(L)
    return W


def weights_from_nanogpt(sd, n_layer):
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    W = dict(wte=sd['transformer.wte.weight'], wpe=sd['transformer.wpe.weight'],
             ln_f=sd['transformer.ln_f.weight'], layers=[])
    for i in range(n_layer):
        p = 'transformer.h.%d.' % i
        W['layers'].append(dict(ln1=sd[p + 'ln_1.weight'], c_attn=sd[p + 'attn.c_attn.weight'],
                                c_proj=sd[p + 'attn.c_proj.weight'], ln2=sd[p + 'ln_2.weight'],
                                c_fc=sd[p + 'mlp.c_fc.weight'], c_mlp=sd[p + 'mlp.c_proj.weight']))
    return W


class Decoder:
    def __init__(self, W, n_layer, n_head, n_embd, Tcap, mode, cfg=None, dev='cpu'):
        self.W, self.nl, self.H, self.C, self.Tcap = W, n_layer, n_head, n_embd, Tcap
        self.dh, self.mode, self.cfg, self.dev = n_embd // n_head, mode, cfg, dev
        self.reset()

    def reset(self):
        if self.mode == 'seg':
            self.states = [SegState(self.cfg, self.Tcap, self.H, self.dh, self.dev) for _ in range(self.nl)]
        else:
            self.states = [DenseState(self.Tcap, self.H, self.dh, self.dev) for _ in range(self.nl)]

    @torch.no_grad()
    def step(self, tok, t):
        """Feed token `tok` at position t. Returns (logits over the vocabulary, average keys read per head/layer)."""
        W, H, C, dh = self.W, self.H, self.C, self.dh
        x = (W['wte'][tok] + W['wpe'][t]).view(1, 1, C)
        keys = 0.0
        for i, L in enumerate(W['layers']):
            q, k, v = F.linear(_ln(x, L['ln1']), L['c_attn']).split(C, -1)
            q, k, v = [a.view(1, 1, H, dh).transpose(1, 2) for a in (q, k, v)]
            if self.mode == 'seg':
                out, kr = seg_attend(self.states[i], L.get('pw'), q, k, v, t)
            else:
                out, kr = self.states[i].attend(q, k, v, t)
            keys += kr
            x = x + F.linear(out.transpose(1, 2).reshape(1, 1, C), L['c_proj'])
            x = x + F.linear(F.gelu(F.linear(_ln(x, L['ln2']), L['c_fc'])), L['c_mlp'])
        return F.linear(_ln(x, W['ln_f']), W['wte'])[0, 0], keys / self.nl
