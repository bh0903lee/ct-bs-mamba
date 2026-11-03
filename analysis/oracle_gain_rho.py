"""Attainable oracle gain of the Pareto-copula generator as a function of its latent dependence rho; writes
results/oracle_gain_vs_rho.json.

For alpha in {2.0, 2.5} and a grid of rho, 300 sequences are simulated as the generator draws them and every
next-event target is scored with the exact conditional law (oracle, grid integration of analysis/oracle_synthetic.py)
and with the closed-form CRPS of the Pareto(alpha, x_min) marginal; the lag-one correlation M of the intervals is
reported on these sequences and on a 3,000-sequence draw. At the two evaluated (alpha, rho) the gain is compared with
results/oracle_synthetic.json (tolerance 0.3 percentage points).
Usage: python analysis/oracle_gain_rho.py [output.json]   (numpy, scipy)
"""
from pathlib import Path
import sys, json, time
import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_synthetic import ParetoCopula, pareto_sequence, pareto_quad, crps_pareto_closed, validate

P = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'results' / 'oracle_gain_vs_rho.json'
REF = P / 'results' / 'oracle_synthetic.json'
ALPHAS = [2.0, 2.5]
XMIN = {2.0: 0.05, 2.5: 0.1}
RHOS = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
N_SEQ, N_SEQ_M = 300, 3000
DATASETS = {(2.5, 0.35): ('syn_corr_pareto_01', 'Pareto–copula 1'), (2.0, 0.45): ('syn_corr_pareto_02', 'Pareto–copula 2')}
TOL_PP = 0.3
UCLIP = 1e-9


def seed_of(alpha, rho, offset=0):
    """Fixed seed per (alpha, rho): 100 alpha * 1000 + 100 rho (+ offset for the M draw)."""
    return int(round(alpha * 100)) * 1000 + int(round(rho * 100)) + offset


def simulate(alpha, rho, n_seq, seed):
    """Sequences in the stored format of the generator (six-decimal intervals and timestamps)."""
    rng = np.random.default_rng(seed)
    xmin, s = XMIN[alpha], np.sqrt(1 - rho ** 2)
    seqs = []
    for _ in range(n_seq):
        L = max(10, int(rng.normal(60, 20)))
        z = np.empty(L); z[0] = rng.standard_normal(); eps = rng.standard_normal(L)
        for t in range(1, L):
            z[t] = rho * z[t - 1] + s * eps[t]
        u = np.clip(ndtr(z), UCLIP, 1 - UCLIP)
        tau = xmin * (1 - u) ** (-1.0 / alpha)
        t_cum = 0.0; seq = []
        for iet in tau:
            t_cum += float(iet)
            seq.append(dict(time_since_start=round(t_cum, 6), time_since_last_event=round(float(iet), 6)))
        seqs.append(seq)
    return seqs


def lag_one_M(seqs):
    """Pearson correlation of adjacent noninitial intervals, pairs within a sequence."""
    a, b = [], []
    for s in seqs:
        x = [e['time_since_last_event'] for e in s[1:]]
        a += x[:-1]; b += x[1:]
    a, b = np.array(a), np.array(b)
    return float(np.mean((a - a.mean()) * (b - b.mean())) / a.std() / b.std())


def evaluate(alpha, rho, rng):
    pc = ParetoCopula(alpha, XMIN[alpha], rho)
    seqs = simulate(alpha, rho, N_SEQ, seed_of(alpha, rho))
    ys, cs, checks, atom, tail, below = [], [], [], [], [], 0
    for si, s in enumerate(seqs):
        y, c, _, tl, xp, nb, at = pareto_sequence(pc, s)
        ys.append(y); cs.append(c); tail.append(tl); atom.append(at); below += nb
        if si < 6:
            i = int(rng.integers(len(y)))
            checks.append(abs(pareto_quad(pc, xp[i], y[i]) / c[i] - 1))
    y = np.concatenate(ys); oracle = np.concatenate(cs)
    marg = crps_pareto_closed(alpha, XMIN[alpha], y)
    seqs_M = simulate(alpha, rho, N_SEQ_M, seed_of(alpha, rho, offset=500))
    return dict(alpha=alpha, rho=rho, xmin=XMIN[alpha], seed=seed_of(alpha, rho), seed_M=seed_of(alpha, rho, 500),
                n_sequences=N_SEQ, n_targets=int(len(y)), zero_targets=int((y == 0).sum()),
                targets_below_xmin=int(below),
                oracle_crps=float(oracle.mean()), marginal_parametric_crps=float(marg.mean()),
                oracle_gain_vs_marginal_percent=float(100 * (oracle.mean() / marg.mean() - 1)),
                oracle_tail_crps=float(oracle[y > np.percentile(y, 90)].mean()),
                marginal_tail_crps=float(marg[y > np.percentile(y, 90)].mean()),
                M_evaluated_sequences=lag_one_M(seqs), M_train_sized_draw=lag_one_M(seqs_M), n_sequences_M=N_SEQ_M,
                oracle_mass_beyond_grid_max=float(np.concatenate(tail).max()),
                upper_clip_atom_mass_max=float(np.concatenate(atom).max()),
                oracle_quad_check_rel_err=float(max(checks)), oracle_quad_checks=len(checks))


def main():
    t0 = time.time()
    val = validate(); print('grid validation', val)
    rng = np.random.default_rng(0)
    ref = json.loads(REF.read_text())
    desc = json.loads((P / 'results' / 'dataset_descriptors.json').read_text())
    rows = []
    for alpha in ALPHAS:
        for rho in RHOS:
            r = evaluate(alpha, rho, rng); rows.append(r)
            print('alpha %.1f rho %.2f  n %5d  oracle %.5f  marg %.5f  gain %6.2f%%  M300 %.3f  M3000 %.3f  quad %.1e' %
                  (alpha, rho, r['n_targets'], r['oracle_crps'], r['marginal_parametric_crps'],
                   r['oracle_gain_vs_marginal_percent'], r['M_evaluated_sequences'], r['M_train_sized_draw'],
                   r['oracle_quad_check_rel_err']), flush=True)
    agreement = {}
    ref_rows = {}
    for (alpha, rho), (ds, name) in DATASETS.items():
        r = next(v for v in rows if v['alpha'] == alpha and v['rho'] == rho)
        g_ref = ref[ds]['oracle_gain_vs_marginal_percent']
        g_ref_par = 100 * (ref[ds]['oracle_crps'] / ref[ds]['marginal_parametric_crps'] - 1)
        ref_rows[ds] = dict(oracle_gain_vs_marginal_percent=g_ref, M=desc[ds]['M'])
        agreement[ds] = dict(alpha=alpha, rho=rho, name=name, simulated_gain_percent=r['oracle_gain_vs_marginal_percent'],
                             dataset_gain_percent_vs_empirical_marginal=g_ref,
                             dataset_gain_percent_vs_parametric_marginal=g_ref_par,
                             difference_pp=r['oracle_gain_vs_marginal_percent'] - g_ref, tolerance_pp=TOL_PP,
                             within_tolerance=bool(abs(r['oracle_gain_vs_marginal_percent'] - g_ref) <= TOL_PP),
                             dataset_M_table2=desc[ds]['M'], simulated_M_evaluated=r['M_evaluated_sequences'],
                             simulated_M_train_sized=r['M_train_sized_draw'])
    out = dict(description=__doc__.strip(), grid_validation=val, alphas=ALPHAS, rhos=RHOS, xmin=XMIN,
               n_sequences=N_SEQ, n_sequences_M=N_SEQ_M, seed_rule=seed_of.__doc__.strip(),
               sequence_length='max(10, int(N(60, 20^2)))', probability_clip=UCLIP,
               marginal='closed-form CRPS of Pareto(alpha, x_min)',
               M='Pearson correlation of adjacent noninitial within-sequence intervals, pooled over sequences',
               agreement_with_oracle_synthetic=agreement, rows=rows, seconds=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print('\n%6s %5s %8s %8s %8s' % ('alpha', 'rho', 'M(300)', 'M(3000)', 'gain%'))
    for r in rows:
        print('%6.1f %5.2f %8.3f %8.3f %8.2f' % (r['alpha'], r['rho'], r['M_evaluated_sequences'],
                                                  r['M_train_sized_draw'], r['oracle_gain_vs_marginal_percent']))
    for ds, a in agreement.items():
        print('%s: simulated %.2f%% vs dataset %.2f%% (diff %+.2f pp, within %.1f pp: %s)' %
              (ds, a['simulated_gain_percent'], a['dataset_gain_percent_vs_empirical_marginal'], a['difference_pp'],
               TOL_PP, a['within_tolerance']))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
