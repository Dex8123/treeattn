import argparse
import json
import os
import pickle

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--out', default='data/tinystories')
p.add_argument('--n_stories', type=int, default=150000)
p.add_argument('--val_stories', type=int, default=2000)
p.add_argument('--vocab', type=int, default=8192)
p.add_argument('--tok_stories', type=int, default=50000)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def take(split, n):
    def loop(ds):
        out = []
        for ex in ds:
            t = ex['text'].strip()
            if len(t) < 50:
                continue
            out.append(t)
            if len(out) >= n:
                break
        return out
    try:
        return loop(load_dataset('roneneldan/TinyStories', split=split, streaming=True))
    except Exception as e:
        print('TinyStories failed (%s); falling back to wikitext-103' % e)
        sp = 'train' if split == 'train' else 'validation'
        return loop(load_dataset('wikitext', 'wikitext-103-raw-v1', split=sp, streaming=True))


train_txt = take('train', args.n_stories)
val_txt = take('validation', args.val_stories)
print('stories: train %d, val %d' % (len(train_txt), len(val_txt)))

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=['<|endoftext|>'],
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tok.train_from_iterator(train_txt[:args.tok_stories], trainer)
tok.save(os.path.join(args.out, 'tokenizer.json'))
eot = tok.token_to_id('<|endoftext|>')


def encode(texts):
    chunks = []
    for i in range(0, len(texts), 5000):
        ids = []
        for enc in tok.encode_batch(texts[i:i + 5000]):
            ids.extend(enc.ids)
            ids.append(eot)
        chunks.append(np.array(ids, dtype=np.uint16))
    return np.concatenate(chunks)


tr = encode(train_txt)
va = encode(val_txt)
tr.tofile(os.path.join(args.out, 'train.bin'))
va.tofile(os.path.join(args.out, 'val.bin'))
vs = tok.get_vocab_size()
with open(os.path.join(args.out, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vs}, f)
stats = {'vocab_size': vs, 'train_tokens': int(len(tr)), 'val_tokens': int(len(va)),
         'train_stories': len(train_txt), 'val_stories': len(val_txt)}
json.dump(stats, open(os.path.join(args.out, 'stats.json'), 'w'))
print(stats)
