"""Per-target analyses of the scored predictions; writes results/prediction_analysis.json.

Per dataset, for the main system (F_ct_bs) and GRU-LogNormMix (seed means unless stated): CRPS, LL and RMSE per target-size decile, share of the total CRPS
in the top decile, the empirical PIT CDF minus u on a 101-point grid for every seed, the overall PIT KS distance per
seed, and the PIT KS distance and central-90% coverage within bins of three history variables (previous-interval
quintile, prefix-position quartile, prefix-burstiness tercile). Scores are recomputed from the mixture parameters.
Inputs: <scores_dir>/<condition>/<dataset>/seed<k>/predictions.npz (evaluation/score_ctbs.py) and
<gru_dir>/<dataset>/seed<k>/predictions.npz (evaluation/train_gru.py).
Usage: python analysis/analyze_predictions.py <data_dir> <scores_dir> <gru_dir> [output.json]
"""
from pathlib import Path
import sys, json, pickle
import numpy as np
P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'evaluation'))
from common import mixture_scores, ks  # noqa: E402
DATA, SCORES, GRU = map(Path, sys.argv[1:4])
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else P / 'results' / 'prediction_analysis.json'
DS = ['syn_exp_01', 'syn_powerlaw_01', 'syn_powerlaw_02', 'syn_corr_pareto_01', 'syn_corr_pareto_02',
      'data_retweet_sampled', 'data_so_sampled', 'data_taxi_sampled', 'data_earthquake', 'data_taobao', 'data_volcano']
MODELS = ['F_ct_bs', 'GRU-LogNormMix']
GRID = np.linspace(0, 1, 101)


def files(model, ds):
    root = GRU / ds if model == 'GRU-LogNormMix' else SCORES / model / ds
    return sorted(root.glob('seed*/predictions.npz'), key=lambda f: int(f.parent.name[4:]))


def history(ds):
    """Per target: previous stored interval, prefix position, burstiness of the intervals seen."""
    p = DATA / ds / ('prezeroclip' if ds == 'syn_exp_01' else '') / 'test.pkl'
    with p.open('rb') as f: o = pickle.load(f, encoding='latin-1')['test']
    ss = list(o.values()) if isinstance(o, dict) else o
    prev, pos, burst = [], [], []
    for s in ss:
        dt = np.array([e['time_since_last_event'] for e in s], float)
        for i in range(1, len(s)):
            prev.append(dt[i - 1]); pos.append(i)
            x = dt[:i]
            if i >= 4:
                cv = x.std() / max(x.mean(), 1e-9); n = i
                burst.append((np.sqrt(n + 1) * cv - np.sqrt(n - 1)) / max((np.sqrt(n + 1) - 2) * cv + np.sqrt(n - 1), 1e-9))
            else: burst.append(0.0)
    return np.array(prev), np.array(pos), np.array(burst)


def bins_of(x, q):
    edges = np.quantile(x, np.linspace(0, 1, q + 1)); b = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, q - 1)
    return b, edges


out = {}
for ds in DS:
    prev, pos, burst = history(ds); ref = None; rec = {}
    hb = {'prev_interval': bins_of(prev, 5), 'position': bins_of(pos, 4), 'burstiness': bins_of(burst, 3)}
    for model in MODELS:
        fs = files(model, ds); assert fs, (model, ds)
        crps, ll, se, curves, kss, cond, cov = [], [], [], [], [], {k: [] for k in hb}, {k: [] for k in hb}
        for f in fs:
            z = np.load(f); y = z['target']
            if ref is None: ref = (z['event_ids'], y); assert len(y) == len(prev)
            assert (z['event_ids'] == ref[0]).all() and (y == ref[1]).all()
            sc = mixture_scores(z['raw'], y); pit = sc['pit']
            crps.append(sc['crps']); ll.append(sc['ll']); se.append((sc['pred'] - y) ** 2)
            srt = np.sort(pit); curves.append((np.searchsorted(srt, GRID, side='right') / len(srt) - GRID).round(5).tolist())
            kss.append(ks(pit))
            for k, (b, _) in hb.items():
                cond[k].append([ks(pit[b == j]) for j in range(b.max() + 1)])
                cov[k].append([float(((pit[b == j] >= .05) & (pit[b == j] <= .95)).mean()) for j in range(b.max() + 1)])
        crps, ll, se = np.mean(crps, 0), np.mean(ll, 0), np.mean(se, 0)
        db, edges = bins_of(y, 10)
        rec[model] = dict(n_seeds=len(fs), decile_crps=[float(crps[db == k].mean()) for k in range(10)],
                          decile_ll=[float(ll[db == k].mean()) for k in range(10)],
                          decile_rmse=[float(np.sqrt(se[db == k].mean())) for k in range(10)],
                          top_decile_crps_share=float(crps[db == 9].sum() / crps.sum()),
                          crps_mean=float(crps.mean()), ll_mean=float(ll.mean()), rmse=float(np.sqrt(se.mean())),
                          pit_curves=curves, ks_per_seed=kss,
                          cond_ks={k: dict(per_seed=v, mean=np.mean(v, 0).tolist(), max_bin=float(np.mean(v, 0).max())) for k, v in cond.items()},
                          cond_cov90={k: np.mean(v, 0).tolist() for k, v in cov.items()})
    out[ds] = dict(n=int(len(ref[1])), decile_edges=np.quantile(ref[1], np.linspace(0, 1, 11)).tolist(), pit_grid=GRID.round(2).tolist(),
                   bin_edges={k: e.tolist() for k, (_, e) in hb.items()}, models=rec)
    print(ds, flush=True)
OUT.write_text(json.dumps(out) + '\n')
print('wrote', OUT)
