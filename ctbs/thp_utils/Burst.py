import numpy as np
import torch
MIN_EVENT = 4

def memory_coefficient_sliding(iet_array):
    # Sliding-window memory coefficient (Pearson r of adjacent IETs) over prefixes of one sequence.
    # A constant window has zero variance, so r is undefined; it is set to 0.0 (no correlation).
    _EPS = 1e-9
    M = dict()
    min_n = MIN_EVENT  # computed from this prefix length onward
    iet_array = iet_array.cpu().numpy()
    for idx_n in range(1, min_n):  # fixed placeholder for prefixes shorter than min_n
        M[idx_n] = 0.2
    for idx_n in range(min_n, iet_array.shape[0]+1):
        iet_seg = iet_array[:idx_n]
        if round(iet_seg[-1], 7) == 0.0:  # padded tail
            for idx in range(idx_n, iet_array.shape[0]+1):
                M[idx] = 0.0
            break
        seg1, seg2 = iet_seg[:-1], iet_seg[1:]
        std1, std2 = seg1.std(), seg2.std()
        if std1 < _EPS or std2 < _EPS:
            # constant window: Pearson r undefined -> 0.0
            M[idx_n] = 0.0
            continue
        inside_sigma = (seg1 - seg1.mean()) * (seg2 - seg2.mean()) / (std1 * std2)
        M[idx_n] = inside_sigma.sum() / seg1.shape[0]
    M = np.array(list(M.values()))
    return M

def burstiness_sliding(iet_array, mode='finite'):
    # Sliding-window burstiness over prefixes of one sequence.
    # Guard: if the window mean is <= EPS, r = std/mean is undefined, so B is set to 0.0.
    _EPS = 1e-9
    FIX_INFTY = 1e-9
    B = dict()
    min_n = MIN_EVENT
    iet_array = iet_array.cpu().numpy()
    if mode == 'finite':
        for idx_n in range(1, min_n):  # 0.0 for prefixes shorter than min_n
            B[idx_n] = 0.0
        for idx_n in range(min_n, iet_array.shape[0]+1):
            iet_seg = iet_array[:idx_n]
            if round(iet_seg[-1], 7) == 0.0:
                for idx in range(idx_n, iet_array.shape[0]+1):
                    B[idx] = 0.0
                break
            mean_iet = iet_seg.mean()
            if mean_iet < _EPS:
                B[idx_n] = 0.0
                continue
            num_event = iet_seg.flatten().shape[0]
            a = np.sqrt(num_event + 1)
            b = np.sqrt(num_event - 1)
            r = iet_seg.std() / mean_iet
            B[idx_n] = (a * r - b) / ((a - 2) * r + b + FIX_INFTY)
    elif mode == 'normal':
        for idx_n in range(1, min_n):  # 0.0 for prefixes shorter than min_n
            B[idx_n] = 0.0
        for idx_n in range(min_n, iet_array.shape[0]+1):
            iet_seg = iet_array[:idx_n]
            if round(iet_seg[-1], 4) == 0.0:
                for idx in range(idx_n, iet_array.shape[0] + 1):
                    B[idx] = 0.0
                break
            mean_iet = iet_seg.mean()
            if mean_iet < _EPS:
                B[idx_n] = 0.0
                continue
            B[idx_n] = (iet_seg.std() - mean_iet) / (iet_seg.std() + mean_iet)
    B = np.array(list(B.values()))
    return B

def memory_coefficient_batch(iet_batch):
    M_dict = dict()
    for idx in range(iet_batch.shape[0]):
        M_dict[idx] = memory_coefficient_sliding(iet_batch[idx])
    M_list = list(M_dict.values())
    M = torch.tensor(M_list, device=iet_batch.device)
    return M

def burstiness_batch(iet_batch):
    B_dict = dict()
    for idx in range(iet_batch.shape[0]):
        B_dict[idx] = burstiness_sliding(iet_batch[idx], 'finite')
    B_list = list(B_dict.values())
    B = torch.tensor(B_list, device=iet_batch.device)
    return B
