"""Memoised Burst.memory_coefficient_batch and Burst.burstiness_batch.

B and M depend only on the data, so the original implementation is cached per sequence row;
results are bitwise identical to the uncached functions.
"""
import torch


def batch_stats(iet_batch):
    """(M, B) for a batch of IETs (B, L), each float64 (B, L).

    CTBS_BM_MODE:
      cache   (default) original implementation, memoised per distinct row
      off     original implementation, recomputed every step
    """
    import os
    import Burst
    mode = os.environ.get('CTBS_BM_MODE', 'cache')
    if mode == 'off':
        return Burst.memory_coefficient_batch(iet_batch), Burst.burstiness_batch(iet_batch)
    if mode != 'cache':
        raise ValueError(f'unknown CTBS_BM_MODE={mode!r} (cache | off)')
    return _cached_original(iet_batch)


# Per-row keys are robust to shuffling; values do not depend on padding length.
_cache = {}
_CACHE_MAX = 40000          # rows


def _cached_original(iet_batch):
    import numpy as np
    import Burst
    dev, dt = iet_batch.device, iet_batch.dtype
    rows = iet_batch.detach().cpu().numpy()
    L = rows.shape[1]
    out_m = np.zeros((len(rows), L), dtype=np.float64)
    out_b = np.zeros((len(rows), L), dtype=np.float64)
    miss = []
    for i, r in enumerate(rows):
        key = r.tobytes()
        hit = _cache.get(key)
        if hit is None:
            miss.append((i, key))
        else:
            out_m[i], out_b[i] = hit
    if miss:
        sub = torch.tensor(rows[[i for i, _ in miss]], dtype=dt)
        m = Burst.memory_coefficient_batch(sub).double().cpu().numpy()
        b = Burst.burstiness_batch(sub).double().cpu().numpy()
        for j, (i, key) in enumerate(miss):
            out_m[i], out_b[i] = m[j], b[j]
            if len(_cache) < _CACHE_MAX:
                _cache[key] = (m[j].copy(), b[j].copy())
    return (torch.tensor(out_m, device=dev),          # float64, as the original returns
            torch.tensor(out_b, device=dev))
