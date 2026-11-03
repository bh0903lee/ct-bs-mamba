"""PIT of the seed-zero main-system models (F_ct_bs) on every dataset; writes results/pit_panel_data.json.gz.
Per dataset: the 20-bin histogram, the empirical CDF (every distinct PIT value with its multiplicity), KS distance
and mean, and the SHA-256 of the scored checkpoint.
Usage: python analysis/export_pit_panels.py <scores_dir> [output.json.gz]   (reads <scores_dir>/F_ct_bs/<dataset>/seed0/)
"""
from pathlib import Path
import sys, json, gzip
import numpy as np
P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'evaluation'))
from common import ks  # noqa: E402
SCORES = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else P / 'results' / 'pit_panel_data.json.gz'
DS = ['syn_exp_01', 'syn_powerlaw_01', 'syn_powerlaw_02', 'syn_corr_pareto_01', 'syn_corr_pareto_02',
      'data_retweet_sampled', 'data_so_sampled', 'data_taxi_sampled', 'data_earthquake', 'data_taobao', 'data_volcano']
edges = np.linspace(0, 1, 21)
rows = []
for ds in DS:
    d = SCORES / 'F_ct_bs' / ds / 'seed0'
    pit = np.load(d / 'predictions.npz')['pit'].astype(float)
    counts, _ = np.histogram(pit, bins=edges)
    v, c = np.unique(pit, return_counts=True)
    rows.append(dict(dataset=ds, n=int(len(pit)), checkpoint_sha256=json.loads((d / 'result.json').read_text())['checkpoint_sha256'],
                     histogram_edges=edges.tolist(), histogram_counts=counts.tolist(), cdf_values=v.tolist(),
                     cdf_counts=c.tolist(), KS=ks(pit), mean=float(pit.mean())))
doc = dict(convention='PIT at the observed target (no target floor), seed-zero F_ct_bs checkpoint, all test targets',
           datasets=rows)
with gzip.GzipFile(OUT, 'wb', mtime=0) as f:
    f.write((json.dumps(doc) + '\n').encode())
print('wrote', OUT)
