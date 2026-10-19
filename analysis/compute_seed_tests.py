"""Seed-level Welch tests on the closed-form CRPS (each pathway condition vs. input only, and
+CT+BS vs. +CT and vs. +BS) for the primary and the NLL-only ablation, and the per-target ratio of the point-loss term to the NLL for the
+CT+BS models.
Usage: python analysis/compute_seed_tests.py [output.json]   (default: results/seed_tests.json)
"""
from pathlib import Path
import csv, json, sys, statistics as st
from scipy import stats
P = Path(__file__).resolve().parents[1]
DS = ['syn_exp_01', 'syn_powerlaw_01', 'syn_powerlaw_02', 'syn_corr_pareto_01', 'syn_corr_pareto_02',
      'data_retweet_sampled', 'data_so_sampled', 'data_taxi_sampled', 'data_earthquake']
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'results' / 'seed_tests.json'
rows = list(csv.DictReader((P / 'results/ablation_crps.csv').open()))
fol = list(csv.DictReader((P / 'results/ablation_eval.csv').open()))
crps = lambda ds, c: [float(r['crps']) for r in rows if r['dataset'] == ds and r['condition'] == c]
out = {'welch': {}, 'point_loss_to_nll': {}}
for ds in DS:
    base = crps(ds, 'input_only')
    out['welch'][ds] = {}
    for c in ['bs', 'ct', 'ct_bs']:
        x = crps(ds, c)
        out['welch'][ds][c] = dict(change_percent=100 * (st.mean(x) / st.mean(base) - 1),
                                   p=float(stats.ttest_ind(x, base, equal_var=False).pvalue))
    for ref in ['ct', 'bs']:  # combination against each single pathway
        x, y = crps(ds, 'ct_bs'), crps(ds, ref)
        out['welch'][ds]['ctbs_vs_' + ref] = dict(change_percent=100 * (st.mean(x) / st.mean(y) - 1),
                                                 p=float(stats.ttest_ind(x, y, equal_var=False).pvalue))
    f = [r for r in fol if r['dataset'] == ds and r['condition'] == 'ct_bs']
    mse = st.mean(float(r['rmse']) ** 2 for r in f); nll = abs(st.mean(float(r['ll']) for r in f))
    out['point_loss_to_nll'][ds] = 1e-3 * mse / nll
# Same tests for the NLL-only ablation (common evaluator; +CT+BS has five seeds).
rec = list(csv.DictReader((P / 'results/nll_ablation_eval.csv').open())); out['welch_nll'] = {}
rc = lambda ds, c: [float(r['crps']) for r in rec if r['dataset'] == ds and r['condition'] == c]
for ds in DS:
    o = out['welch_nll'][ds] = {}
    for c, x, y in [('nll_bs', 'nll_bs', 'nll_input_only'), ('nll_ct', 'nll_ct', 'nll_input_only'),
                    ('nll_ct_bs', 'nll_ct_bs', 'nll_input_only'), ('ctbs_vs_ct', 'nll_ct_bs', 'nll_ct'),
                    ('ctbs_vs_bs', 'nll_ct_bs', 'nll_bs')]:
        a, b = rc(ds, x), rc(ds, y)
        o[c] = dict(change_percent=100 * (st.mean(a) / st.mean(b) - 1), p=float(stats.ttest_ind(a, b, equal_var=False).pvalue))
OUT.write_text(json.dumps(out, indent=2) + '\n')
for ds in DS:
    print(ds, {c: (round(v['change_percent'], 2), round(v['p'], 4)) for c, v in out['welch'][ds].items()},
          'ratio %.3g' % out['point_loss_to_nll'][ds])
