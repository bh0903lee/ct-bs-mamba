"""Dataset descriptors (article Table 2) computed from the train/test splits.
Usage: python analysis/compute_descriptors.py <data_dir> [output.json]
Writes results/dataset_descriptors.json by default; Table 2 of the article reports it.
B and M pool within-sequence intervals of the training split; B is the large-sample (CV-1)/(CV+1);
M is the Pearson correlation of adjacent intervals, pairs never crossing a sequence boundary.
Tail index: generator parameter of the synthetic datasets, read from data/dataset_manifest.json
(Lomax-Hawkes: offspring-delay exponent gamma - 1; Pareto-copula: Pareto shape alpha); for retweet, the Hill
estimate on the largest 5% of positive training intervals (the estimate is threshold dependent).
hill_top5 applies a Hill estimate on the largest 5% of positive training intervals to every dataset (Hill column
of Table 2); zero_test_targets counts the tied events among the test targets as scored by the common evaluator
(float32 timestamp differences).
"""
from pathlib import Path
import sys, json, pickle, statistics as st, math
import numpy as np
P = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'datasets'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else P / 'results' / 'dataset_descriptors.json'
DS = ['syn_exp_01','syn_powerlaw_01','syn_powerlaw_02','syn_corr_pareto_01','syn_corr_pareto_02',
      'data_retweet_sampled','data_so_sampled','data_taxi_sampled','data_earthquake','data_taobao','data_volcano']
MANIFEST = json.loads((P / 'data' / 'dataset_manifest.json').read_text())['datasets']

def load(ds, split):
    with open(DATA / ds / f'{split}.pkl', 'rb') as f:
        d = pickle.load(f, encoding='latin1')[split]
    return list(d.values()) if isinstance(d, dict) else d

def corr(a, b):
    ma, mb = st.fmean(a), st.fmean(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a) / len(a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b) / len(b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a) / sa / sb

def tail_index(ds):
    spec = MANIFEST[ds]
    if spec['kind'] == 'hawkes':
        return round(spec['parameters']['gamma'] - 1, 10), 'generator: Lomax offspring-delay exponent gamma - 1'
    if spec['kind'] == 'pareto':
        return spec['parameters']['alpha'], 'generator: Pareto shape alpha'
    if ds == 'data_retweet_sampled':
        x = sorted((float(e['time_since_last_event']) for s in load(ds, 'train') for e in s[1:]), reverse=True)
        x = [v for v in x if v > 1e-6]; k = int(0.05 * len(x))
        return 1 / st.fmean(math.log(v / x[k]) for v in x[:k]), 'Hill estimate, largest 5% of positive training intervals'
    return None, 'not reported'

out = {}
for ds in DS:
    train, test = load(ds, 'train'), load(ds, 'test')
    iets = [float(e['time_since_last_event']) for s in train for e in s[1:]]
    r = st.pstdev(iets) / st.fmean(iets)
    a, b = [], []
    for s in train:
        x = [float(e['time_since_last_event']) for e in s[1:]]
        a += x[:-1]; b += x[1:]
    targets = [float(e['time_since_last_event']) for s in test for e in s[1:]]
    ti, src = tail_index(ds)
    pos = sorted(x for x in iets if x > 0); k = max(2, int(0.05 * len(pos))); top = pos[-k:]; thr = pos[-k - 1]
    hill = k / sum(math.log(x / thr) for x in top)
    zt = sum(int((np.maximum(np.diff(np.array([e['time_since_start'] for e in s], dtype=np.float32)), 0) == 0).sum())
             for s in test)
    out[ds] = dict(events_per_train_sequence=st.fmean(len(s) for s in train), B=(r - 1) / (r + 1), M=corr(a, b),
                   test_sd=st.stdev(targets), n_test_targets=len(targets), n_train_sequences=len(train),
                   tail_index=ti, tail_index_source=src, hill_top5=hill, zero_test_targets=zt)
OUT.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=1))
