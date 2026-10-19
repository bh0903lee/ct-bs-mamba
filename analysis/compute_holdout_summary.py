"""Summaries of the evaluation on new test data (Supplementary Section S8, Table S10) from
results/holdout_scores.csv and the original per-seed results.
Usage: python analysis/compute_holdout_summary.py [output.json]   (default results/holdout_summary.json)
CRPS changes are between seed means of the same dataset and recipe; Welch tests on per-seed CRPS,
Holm adjustment over the 24 comparisons of each recipe."""
from pathlib import Path
import sys, json, csv, collections
import numpy as np
from scipy import stats
R = Path(__file__).resolve().parents[1] / 'results'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else R / 'holdout_summary.json'
DS = ['syn_exp_01', 'syn_powerlaw_01', 'syn_powerlaw_02', 'syn_corr_pareto_01', 'syn_corr_pareto_02', 'data_retweet_sampled', 'data_so_sampled', 'data_earthquake']
new = collections.defaultdict(list); old = collections.defaultdict(list)
for r in csv.DictReader(open(R / 'holdout_scores.csv')): new[(r['recipe'], r['dataset'], r['condition'])].append(float(r['crps']))
for r in csv.DictReader(open(R / 'ablation_crps.csv')): old[('primary', r['dataset'], r['condition'])].append(float(r['crps']))
for r in csv.DictReader(open(R / 'nll_ablation_eval.csv')): old[('nll_only', r['dataset'], r['condition'])].append(float(r['crps']))
for r in csv.DictReader(open(R / 'common_scoring.csv')):
    if r['model'] == 'GRU-LogNormMix': old[('gru', r['dataset'], 'GRU-LogNormMix')].append(float(r['crps']))
def ch(d, k, b): return 100 * (np.mean(d[k]) / np.mean(d[b]) - 1)
def holm(p):
    o = np.argsort(p); a = np.empty(len(p)); run = 0
    for i, j in enumerate(o): run = max(run, (len(p) - i) * p[j]); a[j] = min(1, run)
    return a
out = {'comparisons': {}, 'system': []}
for fam, pre in [('primary', ''), ('nll_only', 'nll_')]:
    rows = []
    for ds in DS:
        b = (fam, ds, pre + 'input_only')
        for c in ['bs', 'ct', 'ct_bs']:
            k = (fam, ds, pre + c)
            rows.append(dict(dataset=ds, condition=c, holdout_change=ch(new, k, b), original_change=ch(old, k, b),
                             p=float(stats.ttest_ind(new[k], new[b], equal_var=False).pvalue)))
    for r, q in zip(rows, holm(np.array([r['p'] for r in rows]))): r['p_holm'] = float(q)
    out['comparisons'][fam] = rows
for ds in DS:
    g = ('gru', ds, 'GRU-LogNormMix')
    out['system'].append(dict(dataset=ds, full_vs_gru=ch(new, ('primary', ds, 'full_system'), g), full_vs_gru_original=ch(old, ('primary', ds, 'full_system'), g),
                              nll_ctbs_vs_gru=ch(new, ('nll_only', ds, 'nll_ct_bs'), g), nll_ctbs_vs_gru_original=ch(old, ('nll_only', ds, 'nll_ct_bs'), g)))
OUT.write_text(json.dumps(out, indent=1) + '\n')
for fam in out['comparisons']:
    for r in out['comparisons'][fam]: print(fam, r['dataset'], r['condition'], f"{r['holdout_change']:+.2f} ({r['original_change']:+.2f}) holm p={r['p_holm']:.3f}")
