import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from treeattn.config import Config
from treeattn.engine import Decoder, weights_from_nanogpt, weights_from_ours
from treeattn.model import GPT

p = argparse.ArgumentParser()
p.add_argument('--data_dir', default='data/tinystories')
p.add_argument('--ours', required=True)
p.add_argument('--base', required=True)
p.add_argument('--n_windows', type=int, default=6)
p.add_argument('--n_prompts', type=int, default=4)
p.add_argument('--gen_len', type=int, default=100)
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = Tokenizer.from_file(os.path.join(a.data_dir, 'tokenizer.json'))
val = np.memmap(os.path.join(a.data_dir, 'val.bin'), dtype=np.uint16, mode='r')

ck = torch.load(a.ours, map_location=dev, weights_only=False)
cfg = Config(**ck['cfg'])
T = cfg.seq_len
ours = Decoder(weights_from_ours(ck['model'], cfg), cfg.n_layer, cfg.n_head, cfg.n_embd, T, 'seg', cfg, dev)
bk = torch.load(a.base, map_location=dev, weights_only=False)
ma = bk['model_args']
base = Decoder(weights_from_nanogpt(bk['model'], ma['n_layer']), ma['n_layer'], ma['n_head'], ma['n_embd'],
               ma['block_size'], 'dense', None, dev)
rng = np.random.default_rng(7)
starts = rng.integers(0, len(val) - T - 1, size=max(a.n_windows, a.n_prompts))
bytes_per_key = 2 * (cfg.n_embd // cfg.n_head) * cfg.n_head * cfg.n_layer * 4


def score(dec, toks):
    dec.reset()
    tot = keys = 0.0
    for t in range(len(toks) - 1):
        logits, kr = dec.step(int(toks[t]), t)
        tot += F.cross_entropy(logits.unsqueeze(0), torch.tensor([int(toks[t + 1])], device=dev)).item()
        keys += kr
    return tot / (len(toks) - 1), keys / (len(toks) - 1)


def generate(dec, prompt, n):
    dec.reset()
    for t, x in enumerate(prompt[:-1]):
        dec.step(int(x), t)
    seq = [int(x) for x in prompt]
    keys = 0.0
    if dev == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        logits, kr = dec.step(seq[-1], len(seq) - 1)
        seq.append(int(logits.argmax()))
        keys += kr
    if dev == 'cuda':
        torch.cuda.synchronize()
    return seq, keys / n, (time.time() - t0) / n * 1000.0


print('=== 1. Does the inference engine reproduce the training-mode model? (same weights, same window)')
model = GPT(cfg).to(dev).eval()
model.load_state_dict(ck['model'])
w0 = torch.from_numpy(val[starts[0]:starts[0] + T + 1].astype(np.int64)).to(dev)
with torch.no_grad():
    f_loss = model(w0[:T].unsqueeze(0), w0[1:T + 1].unsqueeze(0))[2]['lm'].item()
e_loss, _ = score(ours, w0[:T].tolist())
print('training-mode forward loss %.4f | engine loss %.4f (engine scores 511 of the 512 targets)' % (f_loss, e_loss))

print('\n=== 2. Teacher-forced loss on %d validation windows (causal, every token sees only its past)' % a.n_windows)
res = {'ours': [], 'nanoGPT': []}
kk = {'ours': [], 'nanoGPT': []}
for s in starts[:a.n_windows]:
    w = val[s:s + T].astype(np.int64).tolist()
    for name, dec in (('ours', ours), ('nanoGPT', base)):
        l, k = score(dec, w)
        res[name].append(l)
        kk[name].append(k)
for name in res:
    print('%-8s loss %.4f | keys read per token %.1f | KV bytes read per token %.0f' % (
        name, np.mean(res[name]), np.mean(kk[name]), np.mean(kk[name]) * bytes_per_key))

print('\n=== 3. Greedy generation from %d validation prompts (40 tokens given, %d generated)' % (a.n_prompts, a.gen_len))
stats = {'ours': [], 'nanoGPT': []}
for s in starts[:a.n_prompts]:
    prompt = val[s:s + 40].astype(np.int64).tolist()
    print('\nPROMPT :', tok.decode(prompt).replace('\n', ' '))
    for name, dec in (('ours', ours), ('nanoGPT', base)):
        seq, kr, ms = generate(dec, prompt, a.gen_len)
        gen = seq[len(prompt):]
        tri = [tuple(gen[i:i + 3]) for i in range(len(gen) - 2)]
        rep = 1 - len(set(tri)) / max(len(tri), 1)
        stats[name].append((kr, ms, rep))
        print('%-8s:' % name, tok.decode(gen).replace('\n', ' '))
print()
for name in stats:
    m = np.mean(np.array(stats[name]), 0)
    print('%-8s keys read per generated token %.1f | %.1f ms per token (Python, batch 1) | repeated-3-gram rate %.3f' % (
        name, m[0], m[1], m[2]))
