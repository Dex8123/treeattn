import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import TreeAttention
from .attention_dyn import DynAttention
from .fastseg import SegAttention


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
        self.attn = SegAttention(cfg) if cfg.attn == 'seg' else (DynAttention(cfg) if cfg.attn == 'dyn' else TreeAttention(cfg))
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


def _f(x):
    return float(x.detach()) if torch.is_tensor(x) else float(x)


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
        acc = {}
        for blk in self.blocks:
            x = blk(x)
            for name, val in blk.attn.aux.items():
                acc[name] = acc.get(name, 0.0) + val
        logits = self.lm_head(self.ln_f(x))
        n = len(self.blocks)
        route = acc.pop('route', 0.0)
        pred = acc.pop('pred', 0.0)
        stats = {name: _f(val) / n for name, val in acc.items()}
        if targets is None:
            return logits, None, stats
        lm = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        loss = lm + self.cfg.route_weight * route / n + self.cfg.pred_weight * pred / n
        stats.update(lm=lm.detach(), route=_f(route) / n, pred=_f(pred) / n)
        return logits, loss, stats
