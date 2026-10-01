import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn.config import Config
from treeattn.model import GPT

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
for kw in [dict(), dict(gate=True), dict(beam_mode='pred', global_pool=True)]:
    cfg = Config(**kw)
    torch.manual_seed(0)
    model = GPT(cfg).to(dev).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len), device=dev)
    with torch.no_grad():
        l1, _, _ = model(idx)
        for pos in (70, 200, 333, 500):
            idx2 = idx.clone()
            idx2[:, pos] = (idx[:, pos] + 1) % cfg.vocab_size
            l2, _, _ = model(idx2)
            before = (l1[:, :pos] - l2[:, :pos]).abs().max().item()
            after = (l1[:, pos:] - l2[:, pos:]).abs().max().item()
            print(kw, 'pos', pos, 'change before %.2e after %.2e' % (before, after),
                  'OK' if before < 1e-3 else 'LEAK')
