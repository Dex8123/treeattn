import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import TreeAttention


class LayerNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class MLP(nn.Module):
    """Standard FFN: expand to 4x width, GELU, compress back."""

    def __init__(self, cfg):
        super().__init__()
        C = cfg.n_embd
        self.c_fc = nn.Linear(C, 4 * C, bias=False)
        self.c_proj = nn.Linear(4 * C, C, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = TreeAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """GPT-2 style language model (learned position embeddings, tied output layer) with tree attention."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        b, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        route, pred, kt, ktot = 0.0, 0.0, 0.0, 0.0
        for blk in self.blocks:
            x = blk(x)
            a = blk.attn.aux
            route = route + a.get('route', 0.0)
            pred = pred + a.get('pred', 0.0)
            kt += float(a.get('keys_tree', 0.0))
            ktot += float(a.get('keys_total', 0.0))
        logits = self.lm_head(self.ln_f(x))
        n = len(self.blocks)
        stats = {'keys_tree': kt / n, 'keys_total': ktot / n}
        if targets is None:
            return logits, None, stats
        lm = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        loss = lm + self.cfg.route_weight * route / n + self.cfg.pred_weight * pred / n
        stats.update(lm=lm.detach(), route=float(route) / n, pred=float(pred) / n)
        return logits, loss, stats
