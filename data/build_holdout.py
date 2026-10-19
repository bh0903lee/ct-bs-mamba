#!/usr/bin/env python3
"""Build the new test data of Supplementary Section S8 in a new directory and verify every
dataset against the record hashes in holdout_manifest.json.

Real data (earthquake, retweet, Stack Overflow): source test rows not used by the benchmark, built
with the benchmark conventions of rebuild_datasets.py. Synthetic data: the benchmark generators with
new seeds. The output has the benchmark layout (<dataset>/test.pkl; correlated exponential also
<dataset>/prezeroclip/test.pkl), so evaluation/score_ctbs.py and evaluation/score_gru.py can score
it with --data_dir.
"""
from pathlib import Path
import argparse, json, random, urllib.request
import numpy as np
import pandas as pd
from rebuild_datasets import sha, digest, event, floor, load_pickle, dump_pickle, ordered
from hawkes_generator import generate_dataset
from pareto_generator import gen_dataset
HERE = Path(__file__).resolve().parent


def correlated_exponential(seed, n=1000, test_fraction=0.2):
    """Generator of the correlated-exponential benchmark (theta ~ U(1,100), |kappa| in [0.01,1/3],
    50-500 events, intervals rounded to 0.01); returns the last test_fraction of each sequence."""
    np.random.seed(seed); random.seed(seed)
    mu = np.random.uniform(1, 100, n)
    M = list(np.concatenate((np.random.uniform(-1 / 3, -0.01, n // 2), np.random.uniform(0.01, 1 / 3, n // 2)))); random.shuffle(M)
    ne = [int(x) for x in np.random.uniform(50, 500, n)]
    out = []
    for s in range(n):
        t = [mu[s]]
        for _ in range(1, ne[s]):
            u = np.random.random_sample(1).item(); c = 4 * M[s] * (1 - 2 * np.exp(-t[-1] / mu[s]))
            t.append(mu[s] * np.log(2 * c / (c + 1 - np.sqrt((c + 1) ** 2 - 4 * c * u))))
        iet = [round(x, 2) for x in t]; ts = np.round(np.cumsum(iet), 2)
        ev = [dict(time_since_start=float(a), time_since_last_event=float(b), type_event=0) for a, b in zip(ts, iet)]
        out.append(ev[int(len(ev) * (1 - test_fraction)):])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True, help='download cache for the public source files')
    ap.add_argument('--local-root', type=Path, help='directory holding source files at the manifest local_relative_path entries')
    ap.add_argument('--nhp-so-dir', type=Path, help='canonical NHP data_so/fold1 directory')
    a = ap.parse_args()
    if a.output.exists(): raise SystemExit('Refusing to overwrite an existing output directory')
    bench = json.loads((HERE / 'dataset_manifest.json').read_text())['datasets']
    hold = json.loads((HERE / 'holdout_manifest.json').read_text())['datasets']
    a.cache.mkdir(parents=True, exist_ok=True)

    def source(ds):
        meta = bench[ds]['sources']['test']
        if a.local_root and meta.get('local_relative_path'): p = a.local_root / meta['local_relative_path']
        elif ds == 'data_so_sampled':
            if not a.nhp_so_dir: raise SystemExit('Provide --nhp-so-dir for Stack Overflow fold 1')
            p = a.nhp_so_dir / 'test.pkl'
        else:
            p = a.cache / f'{ds}_test.json'
            if not p.exists():
                with urllib.request.urlopen(meta['url'], timeout=120) as r: p.write_bytes(r.read())
        if sha(p) != meta['sha256']: raise ValueError('Source SHA-256 mismatch: ' + str(p))
        return p

    def write(ds, seqs, sub=''):
        d = a.output / ds / sub; d.mkdir(parents=True, exist_ok=True)
        dump_pickle({'test': dict(enumerate(seqs)), 'dim_process': bench[ds]['dim_process']}, d / 'test.pkl')

    for ds, spec in hold.items():
        if 'rows' in spec:
            p = source(ds)
            if ds == 'data_so_sampled':
                rows = [{k: [e[k] for e in s] for k in ('time_since_start', 'time_since_last_event', 'type_event')} for s in ordered(load_pickle(p), 'test')]
            elif bench[ds]['parser'] == 'pandas.read_json':
                rows = pd.read_json(p).to_dict('records')
            else:
                rows = json.loads(p.read_text())
            shift = bench[ds]['splits']['test']['mark_shift']; seqs = []
            for row in spec['rows']:
                r = rows[row['source_index']]; seq = []
                for j, (t, g, k) in enumerate(zip(r['time_since_start'], r['time_since_last_event'], r['type_event'])):
                    if 'time_offset' in row: t += row['time_offset']; g = row['time_offset'] if j == 0 else g
                    seq.append(event(t, floor(g), k + shift))
                seqs.append(seq)
        elif 'generator_parameters' in spec:
            prm = spec['generator_parameters']
            if bench[ds]['kind'] == 'hawkes':
                seqs = []
                for seq in generate_dataset(**prm)['test']:
                    t = np.asarray([e['time_since_start'] for e in seq]); g = np.diff(t, prepend=0.)
                    seqs.append([event(round(float(tt), 6), floor(round(float(gg), 6)), e['type_event']) for tt, gg, e in zip(t, g, seq)])
            else:
                seqs = ordered(gen_dataset(**prm), 'test')
        else:
            g = spec['generator']; raw = correlated_exponential(g['seed'], g['n_sequences'], g['test_fraction'])
            write(ds, raw, 'prezeroclip')
            seqs = [[event(e['time_since_start'], floor(e['time_since_last_event']), e['type_event']) for e in s] for s in raw]
        if digest(seqs) != spec['record_sha256']: raise ValueError('Record hash differs from holdout_manifest.json: ' + ds)
        write(ds, seqs); print(ds, len(seqs), 'sequences, hash verified', flush=True)


if __name__ == '__main__':
    main()
