"""Seed-level comparisons computed from results/per_seed_scores.csv; writes results/seed_tests.json.

pathway:     per split (test, new_test) and recipe (F, I, P): CRPS change (%) of +BS, +CT, +CT+BS relative to
             input only, Welch p, Holm-adjusted p over the comparisons of the recipe; +CT+BS relative to +CT and +BS
mb_effect:   F relative to I for each condition (Holm over the 44 comparisons)
encoder:     F_input_only_gru relative to F_input_only
system:      +CT+BS of each recipe, F input only and F_input_only_gru relative to GRU-LogNormMix (test split)
new_test_vs_gru: +CT+BS of each recipe relative to GRU-LogNormMix on the new test data and on the test split
recipes:     +CT+BS: P vs I, P_ct_bs_eta0 vs P, F vs P
point_loss_ratio: 1e-3 * seed-mean RMSE^2 / |seed-mean LL| of the P +CT+BS models
validation_ll:    seed-mean validation LL of the +CT+BS models of F, I, P and of the four F conditions
Usage: python analysis/compute_seed_tests.py [output.json]   (needs scipy)
"""
from pathlib import Path
import csv, json, math, sys, statistics as st
from scipy import stats
P = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'results' / 'seed_tests.json'
DS = ['syn_exp_01', 'syn_powerlaw_01', 'syn_powerlaw_02', 'syn_corr_pareto_01', 'syn_corr_pareto_02',
      'data_retweet_sampled', 'data_so_sampled', 'data_taxi_sampled', 'data_earthquake', 'data_taobao', 'data_volcano']
NEW = [d for d in DS if d not in ('data_taxi_sampled', 'data_taobao', 'data_volcano')]
COND = ['input_only', 'bs', 'ct', 'ct_bs']
rows = list(csv.DictReader((P / 'results' / 'per_seed_scores.csv').open()))


def vals(split, rec, cond, ds, m='crps'):
    r = sorted((x for x in rows if (x['split'], x['recipe'], x['condition'], x['dataset']) == (split, rec, cond, ds)),
               key=lambda x: int(x['seed']))
    return [float(x[m]) for x in r if x[m] != '']


def welch(a, b):
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def change(a, b):
    return 100 * (st.fmean(a) / st.fmean(b) - 1)


def cmp(a, b):
    return dict(change_percent=change(a, b), p=welch(a, b), n=len(a), n_ref=len(b))


def with_holm(block):
    for b, q in zip(block, holm([b['p'] for b in block])):
        b['p_holm'] = q
    return block


out = {'pathway': {}}
for split, dss in [('test', DS), ('new_test', NEW)]:
    for rec in 'FIP':
        block = [dict(dataset=ds, condition=c, **cmp(vals(split, rec, c, ds), vals(split, rec, 'input_only', ds)))
                 for ds in dss for c in COND[1:]]
        single = {f'{ds}/ct_bs_vs_{c}': cmp(vals(split, rec, 'ct_bs', ds), vals(split, rec, c, ds))
                  for ds in dss for c in ('ct', 'bs')}
        out['pathway'][f'{split}/{rec}'] = dict(comparisons=with_holm(block), ct_bs_vs_single=single)
out['mb_effect'] = with_holm([dict(dataset=ds, condition=c, **cmp(vals('test', 'F', c, ds), vals('test', 'I', c, ds)))
                              for ds in DS for c in COND])
out['encoder'] = {ds: cmp(vals('test', 'F', 'input_only_gru', ds), vals('test', 'F', 'input_only', ds)) for ds in DS}
gru = lambda split, ds: vals(split, 'GRU-LogNormMix', 'reference', ds)
out['system'] = {ds: {k: cmp(vals('test', r, c, ds), gru('test', ds)) for k, r, c in
                      [('F_ct_bs', 'F', 'ct_bs'), ('I_ct_bs', 'I', 'ct_bs'), ('P_ct_bs', 'P', 'ct_bs'),
                       ('F_input_only', 'F', 'input_only'), ('F_input_only_gru', 'F', 'input_only_gru')]} for ds in DS}
out['new_test_vs_gru'] = {ds: {r: dict(new_test=cmp(vals('new_test', r, 'ct_bs', ds), gru('new_test', ds)),
                                       test=cmp(vals('test', r, 'ct_bs', ds), gru('test', ds))) for r in 'FIP'} for ds in NEW}
out['recipes'] = {ds: {k: cmp(vals('test', *a, ds), vals('test', *b, ds)) for k, a, b in
                       [('P_vs_I', ('P', 'ct_bs'), ('I', 'ct_bs')), ('P_eta0_vs_P', ('P', 'ct_bs_eta0'), ('P', 'ct_bs')),
                        ('F_vs_P', ('F', 'ct_bs'), ('P', 'ct_bs'))]} for ds in DS}
out['point_loss_ratio'] = {ds: 1e-3 * st.fmean(x ** 2 for x in vals('test', 'P', 'ct_bs', ds, 'rmse'))
                           / abs(st.fmean(vals('test', 'P', 'ct_bs', ds, 'll'))) for ds in DS}
out['validation_ll'] = {ds: dict(ct_bs={r: st.fmean(vals('test', r, 'ct_bs', ds, 'val_ll')) for r in 'FIP'},
                                 F={c: st.fmean(vals('test', 'F', c, ds, 'val_ll')) for c in COND}) for ds in DS}
OUT.write_text(json.dumps(out, indent=1) + '\n')
print('wrote', OUT)
