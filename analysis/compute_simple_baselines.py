"""CRPS of two history-light reference forecasts on the common next-event targets.
  marginal:     empirical distribution of all noninitial training intervals (no history).
  running mean: exponential distribution whose mean is the average of the observed intervals so far.
Usage: python analysis/compute_simple_baselines.py <data_dir> [output.json]
Writes results/simple_baselines.json by default.
"""
from pathlib import Path
import sys, json, pickle
import numpy as np
P = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'datasets'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else P / 'results' / 'simple_baselines.json'
DS = ['syn_exp_01', 'syn_powerlaw_01', 'syn_powerlaw_02', 'syn_corr_pareto_01', 'syn_corr_pareto_02',
      'data_retweet_sampled', 'data_so_sampled', 'data_taxi_sampled', 'data_earthquake']


def load(ds, split):
    with open(DATA / ds / f'{split}.pkl', 'rb') as f:
        d = pickle.load(f, encoding='latin1')[split]
    return list(d.values()) if isinstance(d, dict) else d


def crps_empirical(sample, y):
    """Exact CRPS of the empirical distribution of `sample` at each y."""
    x = np.sort(sample); n = len(x); c = np.cumsum(x)
    half_mean_abs_diff = np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2  # = E|X-X'| / 2
    k = np.searchsorted(x, y, side='right')
    below = k * y - c[k - 1].clip(min=0) * (k > 0)
    above = (c[-1] - np.where(k > 0, c[k - 1], 0)) - (n - k) * y
    return (below + above) / n - half_mean_abs_diff


out = {}
for ds in DS:
    train = load(ds, 'train')
    pool = np.array([float(e['time_since_last_event']) for s in train for e in s[1:]])
    y_all, rm = [], []
    for s in load(ds, 'test'):
        t = np.array([e['time_since_start'] for e in s], dtype=np.float32)
        y = np.maximum(np.diff(t), 0).astype(float)          # same targets as the common evaluator
        stored = np.array([float(e['time_since_last_event']) for e in s])
        for i in range(1, len(s)):
            m = max(stored[:i].mean(), 1e-6)
            rm.append(y[i - 1] + 2 * m * np.exp(-y[i - 1] / m) - 1.5 * m)  # exponential CRPS
            y_all.append(y[i - 1])
    y_all = np.array(y_all)
    out[ds] = dict(n_targets=len(y_all), marginal_crps=float(crps_empirical(pool, y_all).mean()),
                   running_mean_exponential_crps=float(np.mean(rm)))
    print(ds, out[ds], flush=True)
OUT.write_text(json.dumps(out, indent=2) + '\n')
