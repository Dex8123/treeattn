"""Synthetic non-local recall blocks, built from ordinary text blocks (no model code, nothing quadratic).

A block of B text tokens gets n needles "MARK key value" at random places and, at its end, n queries
"QRY key value" in shuffled order. To answer a query the model must find the needle with the same key,
anywhere earlier in the block (up to ~B tokens away) and copy its value. Rare token ids in the text are
shifted away so the marker, key and value ids only occur in needles and queries.
"""
import numpy as np

NK = 500  # size of the key pool and of the value pool


def ids(V):
    return dict(KEY0=V - 1100, VAL0=V - 600, MARK=V - 100, QRY=V - 99)


def make_recall_block(text, V, rng, n=16):
    """text: int array of length B. Returns tokens (B,), query_pos (n,) = position of each query's KEY token
    (the model predicts the value at the next position), needle_pos (n,) = where that key's needle starts,
    values (n,) = the correct value token ids."""
    B = len(text)
    I = ids(V)
    x = np.where(text >= V - 1100, text - 1100, text).astype(np.int64)
    n_slots = (B - 4 * n - 8) // 4
    slots = rng.choice(n_slots, n, replace=False)
    keys = rng.choice(NK, n, replace=False)
    vals = rng.integers(0, NK, n)
    needle_pos = slots * 4
    for p, kk, vv in zip(needle_pos, keys, vals):
        x[p], x[p + 1], x[p + 2] = I['MARK'], I['KEY0'] + kk, I['VAL0'] + vv
    order = rng.permutation(n)
    qpos = np.zeros(n, dtype=np.int64)
    base = B - 3 * n
    for j, o in enumerate(order):
        p = base + 3 * j
        x[p], x[p + 1], x[p + 2] = I['QRY'], I['KEY0'] + keys[o], I['VAL0'] + vals[o]
        qpos[j] = p + 1
    return x, qpos, needle_pos[order], I['VAL0'] + vals[order]
