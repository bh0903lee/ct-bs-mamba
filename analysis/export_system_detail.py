"""Per-target summaries of the GRU-LogNormMix reference and the full CT-BS system (decile CRPS and PIT curves, Figure 3).
Reads the predictions.npz files written by evaluation/train_gru.py and evaluation/score_ctbs.py,
organized as <dir>/<dataset>/seed<k>/predictions.npz, and writes results/system_detail.json:
mean CRPS per target-size decile (averaged over seeds) and the empirical PIT CDF minus u on a
101-point grid for every seed, keyed 'gru' and 'full_system'.
Usage: python analysis/export_system_detail.py <gru_dir> <full_system_dir> [output_json]
"""
from pathlib import Path
import sys, json
import numpy as np
P = Path(__file__).resolve().parents[1]
GRU, FULL = Path(sys.argv[1]), Path(sys.argv[2])
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else P / 'results' / 'system_detail.json'
DS = ['syn_exp_01', 'syn_powerlaw_01', 'syn_powerlaw_02', 'syn_corr_pareto_01', 'syn_corr_pareto_02',
      'data_retweet_sampled', 'data_so_sampled', 'data_taxi_sampled', 'data_earthquake']
GRID = np.linspace(0, 1, 101)
out = {}
for ds in DS:
    rec, ref = {}, None
    for name, root, n in [('gru', GRU, 5), ('full_system', FULL, 10)]:
        fs = sorted((root / ds).glob('seed*/predictions.npz')); assert len(fs) == n, (ds, name, len(fs))
        crps, curves = [], []
        for f in fs:
            z = np.load(f)
            if ref is None: ref = (z['event_ids'], z['target'])
            assert (z['event_ids'] == ref[0]).all() and (z['target'] == ref[1]).all()
            crps.append(z['crps'])
            pit = np.sort(z['pit'])
            curves.append((np.searchsorted(pit, GRID, side='right') / len(pit) - GRID).round(5).tolist())
        rec[name] = dict(crps=np.mean(crps, axis=0), pit=curves)
    y = ref[1]; edges = np.quantile(y, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, y, side='right') - 1, 0, 9)
    out[ds] = dict(n=int(len(y)), decile_edges=edges.tolist(),
                   decile_crps={k: [float(v['crps'][bins == j].mean()) for j in range(10)] for k, v in rec.items()},
                   pit_grid=GRID.round(2).tolist(), pit={k: v['pit'] for k, v in rec.items()})
OUT.write_text(json.dumps(out) + '\n')
print('wrote', OUT)
