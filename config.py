from dataclasses import dataclass, asdict


@dataclass
class Config:
    vocab_size: int = 8192
    seq_len: int = 512          # tokens per training window (power of two)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    attn: str = 'block'         # 'block' = beta 1 (one tree per 64-token block), 'dyn' = round 4 dynamic single tree
    # ---- beta 1 block attention
    tree_block: int = 64
    # ---- shared search settings
    wmax: int = 8               # max branches kept per tree level (max leaves found per query)
    beam: int = 4               # leaf budget when beam_mode == 'fixed'
    beam_mode: str = 'fixed'    # 'fixed' or 'pred' (learned budget head)
    beam_classes: tuple = (2, 3, 4, 6, 8)   # leaf budgets the head chooses between
    global_pool: bool = False   # budget head also sees the mean of all earlier keys
    gate: bool = False          # soft mean-log dilution gate (block attention only; off)
    gate_q: float = 0.25
    gate_min_level: int = 0     # seg: apply the gate only at tree levels >= this (0 = all levels)
    gate_temp: float = 0.3
    kmeans_iters: int = 3
    route_weight: float = 0.0   # routing loss: switched off after the round 3 results
    pred_weight: float = 0.1
    # ---- round 4 dynamic tree attention
    chunk: int = 64             # training lane length: the sequence is cut into lanes searched in parallel
    local: int = 3              # static recent keys (t, t-1, t-2 for 3); a larger value adds an exact local window
    neighbors: bool = True      # neighbour search on/off
    nb_window: int = 8          # how many earlier queries a query may borrow groups from
    nb_store: int = 2           # how many best leaf groups each query stores for its neighbours
    nb_rho: float = 0.35        # a stored group matches if the new score >= old score - rho * |old score|
    nb_probe: int = 3           # nearest neighbours used to estimate the match rate (sets the variable cap)
    nb_cap_min: int = 2         # smallest neighbour scan width

    def to_dict(self):
        return asdict(self)
