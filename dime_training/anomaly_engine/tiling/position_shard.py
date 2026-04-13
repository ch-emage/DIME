# anomaly_engine/tiling/position_shard.py
from typing import List, Tuple

def enumerate_positions(rows: int, cols: int) -> List[Tuple[int,int,int]]:
    """
    Returns [(pos_id, r, c)] row-major. pos_id = r*cols + c
    """
    out = []
    pid = 0
    for r in range(rows):
        for c in range(cols):
            out.append((pid, r, c))
            pid += 1
    return out

def shard_positions(rows: int, cols: int, world_size: int, rank: int) -> List[int]:
    """
    Assign pos_id to ranks by (pos_id % world_size == rank).
    """
    poss = enumerate_positions(rows, cols)
    return [pid for (pid, _, _) in poss if (pid % world_size) == rank]
