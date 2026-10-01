from dataclasses import dataclass, asdict


@dataclass
class Config:
    vocab_size: int = 8192
    seq_len: int = 512          # tokens per training window (must be a multiple of tree_block)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    tree_block: int = 64        # keys per block; every completed block gets its own tree (power of two)
    wmax: int = 8               # max branches kept per tree level
    beam: int = 4               # branches kept per level in 'fixed' mode
    beam_mode: str = 'fixed'    # 'fixed' or 'pred' (learned budget head)
    beam_classes: tuple = (2, 3, 4, 6, 8)   # beam sizes the budget head chooses between
    global_pool: bool = False   # budget head also sees the pooled keys of earlier blocks (untested idea)
    gate: bool = False          # soft mean-log dilution gate on the index vectors
    gate_q: float = 0.25
    gate_temp: float = 0.3
    kmeans_iters: int = 3
    route_weight: float = 0.05  # weight of the auxiliary routing loss
    pred_weight: float = 0.1    # weight of the budget-head loss

    def to_dict(self):
        return asdict(self)
