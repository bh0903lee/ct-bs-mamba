"""Oracle CRPS on the synthetic test targets whose generator is known; writes results/oracle_synthetic.json.

Scores every common-evaluator target with the conditional law implied by the generator (Lomax-Hawkes given the
stored history; Pareto-copula given the previous interval), the empirical marginal of the noninitial training
intervals and a parametric marginal; for the correlated-exponential data (no stored per-sequence parameters) only
the marginal and running-mean exponential references. CRPS is integrated numerically on per-target log grids,
checked against closed forms and scipy.integrate.quad.
Usage: python analysis/oracle_synthetic.py <data_dir> [output.json]   (numpy, scipy)
"""
from pathlib import Path
import sys, json, pickle, time
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.integrate import quad

P = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'datasets'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else P / 'results' / 'oracle_synthetic.json'
EXPECTED_N = dict(syn_powerlaw_01=25682, syn_powerlaw_02=25895, syn_corr_pareto_01=17686,
                  syn_corr_pareto_02=16960, syn_exp_01=52489)
HAWKES = dict(syn_powerlaw_01=dict(lam0=0.3, nstar=0.7, omega=1.0, gamma=2.2),
              syn_powerlaw_02=dict(lam0=0.2, nstar=0.65, omega=0.5, gamma=3.2))
PARETO = dict(syn_corr_pareto_01=dict(alpha=2.5, xmin=0.1, rho=0.35),
              syn_corr_pareto_02=dict(alpha=2.0, xmin=0.05, rho=0.45))
UCLIP = 1e-9
G_IN = np.logspace(-9, 0, 1001)            # fraction of the inner segment
LOG_PER_DECADE = 125


def load(ds, split):
    with open(DATA / ds / f'{split}.pkl', 'rb') as f:
        d = pickle.load(f, encoding='latin-1')[split]
    return list(d.values()) if isinstance(d, dict) else d


def targets_of(seq):
    """Common-evaluator targets: float32 timestamp differences, clamped at 0, cast to float64."""
    t = np.array([e['time_since_start'] for e in seq], dtype=np.float32)
    return np.maximum(t[1:] - t[:-1], 0).astype(np.float64)


def crps_empirical(sample, y):
    """Exact CRPS of the empirical distribution of `sample` at each y."""
    x = np.sort(sample); n = len(x); c = np.cumsum(x)
    half_mean_abs_diff = np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2
    k = np.searchsorted(x, y, side='right')
    below = k * y - c[k - 1].clip(min=0) * (k > 0)
    above = (c[-1] - np.where(k > 0, c[k - 1], 0)) - (n - k) * y
    return (below + above) / n - half_mean_abs_diff


def crps_exponential(m, y):
    return y + 2 * m * np.exp(-y / m) - 1.5 * m


def crps_pareto_closed(alpha, xmin, y):
    """CRPS of Pareto(alpha, xmin), F(t) = 1 - (t/xmin)^-alpha for t >= xmin."""
    y = np.asarray(y, dtype=np.float64)
    r = np.maximum(y / xmin, 1.0)
    inner = xmin * ((r - 1) - 2 * (r ** (1 - alpha) - 1) / (1 - alpha) + (r ** (1 - 2 * alpha) - 1) / (1 - 2 * alpha))
    outer = xmin * r ** (1 - 2 * alpha) / (2 * alpha - 1)
    return np.maximum(xmin - y, 0) + inner + outer


def outer_grid(lo, vmax):
    """Offsets v (log-spaced from lo to vmax) for the segment [y, y + vmax]."""
    n = int(np.ceil(np.log10(vmax / lo) * LOG_PER_DECADE)) + 1
    return np.logspace(np.log10(lo), np.log10(vmax), n)


def crps_grid(F, y, lower, vmax, lo_abs):
    """int_lower^y F^2 dt + int_y^{y+vmax} (1-F)^2 dt for CDFs F(t) of shape (n, K) on t (n, K).
    `lower` is the support start (0 for Hawkes, xmin for Pareto): F == 0 below it. Returns
    (crps_without_the_piece_below_lower, 1 - F at the last grid point)."""
    y = np.asarray(y, dtype=np.float64)
    yl = np.maximum(y, lower)
    t_in = lower + (yl - lower)[:, None] * G_IN[None, :]
    v = outer_grid(lo_abs, vmax)
    t_out = yl[:, None] + v[None, :]
    f_in = F(t_in) ** 2
    s_out = 1 - F(t_out)
    inner = np.trapz(f_in, t_in, axis=1)
    outer = np.trapz(s_out ** 2, t_out, axis=1)
    return inner + outer, s_out[:, -1]


class Hawkes:
    def __init__(self, lam0, nstar, omega, gamma):
        self.lam0, self.nstar, self.omega, self.gamma = lam0, nstar, omega, gamma

    def Phi(self, v):
        return self.nstar * (1 - (1 + self.omega * v) ** (1 - self.gamma))

    def phi(self, v):
        return self.nstar * (self.gamma - 1) * self.omega * (1 + self.omega * v) ** (-self.gamma)

    def Lambda(self, d, w, u):
        """Compensator of the next interval: d (n, L) gaps T - t_j, w (n, L) history mask, u (n, K)."""
        inc = self.Phi(d[:, None, :] + u[:, :, None]) - self.Phi(d)[:, None, :]
        return self.lam0 * u + np.einsum('nkl,nl->nk', inc, w)

    def intensity(self, d, w, u):
        return self.lam0 + np.einsum('nkl,nl->nk', self.phi(d[:, None, :] + u[:, :, None]), w)


def hawkes_sequence(h, seq, vmax=1e3, chunk_elems=3e7):
    """Oracle CRPS, log density and tail mass beyond the grid for every target of one sequence."""
    t = np.array([e['time_since_start'] for e in seq], dtype=np.float64)
    y = targets_of(seq)
    L = len(t); n = L - 1
    idx = np.arange(1, L)
    D = t[idx - 1][:, None] - t[None, :]
    W = (np.arange(L)[None, :] < idx[:, None]).astype(np.float64)
    D = np.where(W > 0, D, 0.0)
    crps = np.empty(n); ll = np.empty(n); tail = np.empty(n)
    K = len(G_IN) + len(outer_grid(1e-9, vmax))
    step = max(1, int(chunk_elems // (K * L)))
    for a in range(0, n, step):
        b = min(n, a + step); d = D[a:b]; w = W[a:b]; yy = y[a:b]
        F = lambda u: -np.expm1(-h.Lambda(d, w, u))
        c, s_last = crps_grid(F, yy, 0.0, vmax, 1e-9)
        crps[a:b] = c; tail[a:b] = s_last
        yf = np.maximum(yy, 1e-6)[:, None]
        ll[a:b] = (np.log(h.intensity(d, w, yf)) - h.Lambda(d, w, yf))[:, 0]
    return y, crps, ll, tail, D, W


def hawkes_quad(h, d, w, y, vmax=1e3):
    """scipy.integrate.quad reference for one target (d, w: (L,) history)."""
    F = lambda u: float(-np.expm1(-h.Lambda(d[None], w[None], np.array([[u]]))[0, 0]))
    a = quad(lambda u: F(u) ** 2, 0, y, limit=200)[0] if y > 0 else 0.0
    b = quad(lambda u: (1 - F(u)) ** 2, y, y + vmax, limit=400, points=[y + 1e-3, y + 1e-1, y + 1, y + 10, y + 100])[0]
    return a + b


class ParetoCopula:
    def __init__(self, alpha, xmin, rho):
        self.alpha, self.xmin, self.rho, self.s = alpha, xmin, rho, np.sqrt(1 - rho ** 2)
        self.tau_max = xmin * UCLIP ** (-1 / alpha)

    def xi(self, tau):
        """Gaussian score of an interval: xi = Phi^-1(1 - (tau/xmin)^-alpha), survival-side clipping
        exactly as the generator clips u to [1e-9, 1-1e-9]."""
        sv = np.exp(-self.alpha * np.log(np.maximum(tau / self.xmin, 1.0)))   # 1 - u
        sv = np.clip(sv, UCLIP, 1 - UCLIP)
        return np.where(sv < 0.5, -ndtri(sv), ndtri(1 - sv))

    def F(self, t, xi_prev):
        """Conditional CDF of the next interval given the previous one; (n, K) on t with xi_prev (n,)."""
        z = (self.xi(t) - self.rho * xi_prev[:, None]) / self.s
        return np.where(t >= self.tau_max, 1.0, np.where(t >= self.xmin, ndtr(z), 0.0))

    def upper_atom(self, xi_prev):
        """Mass the generator places at tau_max (u clipped at 1 - 1e-9) given the predecessor."""
        return 1 - ndtr((ndtri(1 - UCLIP) - self.rho * xi_prev) / self.s)

    def logpdf(self, t, xi_prev):
        xi = self.xi(t); z = (xi - self.rho * xi_prev) / self.s
        return (-0.5 * z ** 2 - np.log(self.s) + np.log(self.alpha / self.xmin)
                - (self.alpha + 1) * np.log(t / self.xmin) + 0.5 * xi ** 2)   # log phi(z) - log phi(xi) cancels the 2pi


def pareto_sequence(pc, seq, vmul=1e8):
    tau_stored = np.array([e['time_since_last_event'] for e in seq], dtype=np.float64)
    y = targets_of(seq)
    xi_prev = pc.xi(tau_stored[:-1])
    F = lambda t: pc.F(t, xi_prev)
    c, s_last = crps_grid(F, y, pc.xmin, vmul * pc.xmin, 1e-9 * pc.xmin)
    crps = c + np.maximum(pc.xmin - y, 0)
    yf = np.maximum(np.maximum(y, 1e-6), pc.xmin)
    ll = pc.logpdf(yf, xi_prev)
    return y, crps, ll, s_last, xi_prev, int((y < pc.xmin).sum()), pc.upper_atom(xi_prev)


def pareto_quad(pc, xi_prev, y, vmul=1e8):
    F = lambda t: float(pc.F(np.array([[t]]), np.array([xi_prev]))[0, 0])
    yl = max(y, pc.xmin)
    a = quad(lambda t: F(t) ** 2, pc.xmin, yl, limit=200)[0] if yl > pc.xmin else 0.0
    pts = [yl * 10 ** k for k in range(1, 8)]
    b = quad(lambda t: (1 - F(t)) ** 2, yl, yl + vmul * pc.xmin, limit=400, points=pts)[0]
    return a + b + max(pc.xmin - y, 0)


def validate():
    """Grid integrator vs closed forms (exponential, Pareto) and vs quad (Pareto)."""
    ys = np.array([1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0])
    m = 1.3
    F = lambda t: 1 - np.exp(-t / m)
    g, _ = crps_grid(F, ys, 0.0, 1e3, 1e-9)
    exp_err = float(np.max(np.abs(g / crps_exponential(m, ys) - 1)))
    alpha, xmin = 2.5, 0.1
    ys = np.array([0.0999, 0.1, 0.100001, 0.11, 0.15, 0.3, 1.0, 5.0])
    F = lambda t: np.where(t >= xmin, 1 - (np.maximum(t, xmin) / xmin) ** (-alpha), 0.0)
    g, _ = crps_grid(F, ys, xmin, 1e8 * xmin, 1e-9 * xmin)
    g = g + np.maximum(xmin - ys, 0)
    closed = crps_pareto_closed(alpha, xmin, ys)
    q = []
    for y in ys:
        yl = max(y, xmin); f = lambda t: 1 - (t / xmin) ** (-alpha)
        a = quad(lambda t: f(t) ** 2, xmin, yl)[0] if yl > xmin else 0.0
        b = quad(lambda t: (1 - f(t)) ** 2, yl, np.inf)[0]
        q.append(a + b + max(xmin - y, 0))
    q = np.array(q)
    return dict(exponential_check_rel_err=exp_err,
                pareto_check_rel_err=float(np.max(np.abs(g / closed - 1))),
                pareto_quad_vs_closed_rel_err=float(np.max(np.abs(q / closed - 1))))


def deciles(y, crps, ref):
    edges = np.quantile(y, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges[1:-1], y, side='right'), 0, 9)
    return dict(edges=[float(e) for e in edges],
                oracle_crps=[float(crps[bins == k].mean()) for k in range(10)],
                marginal_empirical_crps=[float(ref[bins == k].mean()) for k in range(10)],
                n=[int((bins == k).sum()) for k in range(10)])


def main():
    rng = np.random.default_rng(0)
    val = validate()
    print('validation', val)
    out = dict(validation=val, grid=dict(inner_points=len(G_IN), outer_points_per_decade=LOG_PER_DECADE))
    rows = []
    for ds in ['syn_powerlaw_01', 'syn_powerlaw_02', 'syn_corr_pareto_01', 'syn_corr_pareto_02', 'syn_exp_01']:
        t0 = time.time()
        train = load(ds, 'train'); test = load(ds, 'test')
        pool = np.array([float(e['time_since_last_event']) for s in train for e in s[1:]])
        y_all = np.concatenate([targets_of(s) for s in test])
        n = len(y_all); assert n == EXPECTED_N[ds], (ds, n, EXPECTED_N[ds])
        marg = crps_empirical(pool, y_all)
        rec = dict(n_targets=n, n_targets_expected=EXPECTED_N[ds], zero_targets=int((y_all == 0).sum()),
                   marginal_empirical_crps=float(marg.mean()))
        if ds in HAWKES:
            h = Hawkes(**HAWKES[ds]); crps = []; ll = []; tail = []; checks = []
            for si, s in enumerate(test):
                y, c, l, tl, D, W = hawkes_sequence(h, s)
                crps.append(c); ll.append(l); tail.append(tl)
                if si < 12:
                    i = int(rng.integers(len(y)))
                    checks.append(abs(hawkes_quad(h, D[i], W[i], y[i]) / c[i] - 1))
            crps = np.concatenate(crps); ll = np.concatenate(ll); tail = np.concatenate(tail)
            rate = h.lam0 / (1 - h.nstar)
            rec.update(oracle_crps=float(crps.mean()), oracle_ll=float(ll.mean()),
                       marginal_parametric_crps=float(crps_exponential(1 / rate, y_all).mean()),
                       marginal_parametric='exponential with the stationary rate lam0/(1-n*) = %.6g' % rate,
                       oracle_mass_beyond_grid_max=float(tail.max()), oracle_mass_beyond_grid_mean=float(tail.mean()),
                       oracle_quad_check_rel_err=float(max(checks)), oracle_quad_checks=len(checks),
                       oracle_tail_crps=float(crps[y_all > np.percentile(y_all, 90)].mean()),
                       marginal_empirical_tail_crps=float(marg[y_all > np.percentile(y_all, 90)].mean()),
                       assumptions='Lomax-Hawkes conditional law given the stored events: lambda(t) = lam0 + sum phi(t - t_j) over '
                                   'all stored events before the target; the stored events are treated as the full history '
                                   '(the generator centers timestamps at the first simulated event, so no earlier events exist). '
                                   'The finite simulation horizon (T_max = 50/80), the 5-500 length filter and the six-decimal '
                                   'rounding are ignored, i.e. the oracle is the uncensored conditional law of the process. '
                                   'History timestamps use the stored float64 values; targets follow the common evaluator '
                                   '(float32 differences). LL uses log density at max(y, 1e-6).')
        elif ds in PARETO:
            pc = ParetoCopula(**PARETO[ds]); crps = []; ll = []; tail = []; below = 0; checks = []; atom = []
            for si, s in enumerate(test):
                y, c, l, tl, xp, nb, at = pareto_sequence(pc, s)
                crps.append(c); ll.append(l); tail.append(tl); below += nb; atom.append(at)
                if si < 12:
                    i = int(rng.integers(len(y)))
                    checks.append(abs(pareto_quad(pc, xp[i], y[i]) / c[i] - 1))
            crps = np.concatenate(crps); ll = np.concatenate(ll); tail = np.concatenate(tail); atom = np.concatenate(atom)
            rec.update(oracle_crps=float(crps.mean()), oracle_ll=float(ll.mean()),
                       marginal_parametric_crps=float(crps_pareto_closed(pc.alpha, pc.xmin, y_all).mean()),
                       marginal_parametric='Pareto(alpha=%g, xmin=%g) closed form' % (pc.alpha, pc.xmin),
                       oracle_mass_beyond_grid_max=float(tail.max()), targets_below_xmin=int(below),
                       upper_clip_tau_max=float(pc.tau_max), upper_clip_atom_mass_max=float(atom.max()),
                       upper_clip_atom_mass_mean=float(atom.mean()),
                       oracle_quad_check_rel_err=float(max(checks)), oracle_quad_checks=len(checks),
                       oracle_tail_crps=float(crps[y_all > np.percentile(y_all, 90)].mean()),
                       marginal_empirical_tail_crps=float(marg[y_all > np.percentile(y_all, 90)].mean()),
                       assumptions='Pareto-copula conditional law given the previous interval: the Gaussian score of the '
                                   'predecessor is recovered from the stored time_since_last_event of event i-1 (for the target '
                                   'at i = 1 this is the first stored interval, which equals the first generator draw), with the '
                                   "generator's probability clipping [1e-9, 1-1e-9] applied to the survival value; the "
                                   'six-decimal rounding of the stored predecessor is ignored (it perturbs the recovered score '
                                   'slightly, so the oracle is an infinitesimally noisy version of the exact one). The '
                                   'conditional CDF is F(t) = 0 below xmin, carries the generator\'s clipping atoms at xmin '
                                   '(u < 1e-9) and at tau_max = xmin * 1e-9^(-1/alpha) (u > 1 - 1e-9; conditional mass '
                                   'reported as upper_clip_atom_mass), and equals 1 at and beyond tau_max; the six-decimal '
                                   'rounding of the target is ignored. Targets below xmin (float32 rounding) contribute '
                                   'xmin - y to the CRPS; LL uses log density at max(y, 1e-6, xmin).')
        else:
            rm = []
            for s in test:
                y = targets_of(s); stored = np.array([float(e['time_since_last_event']) for e in s])
                cm = np.cumsum(stored)[:-1] / np.arange(1, len(s))
                rm.append(crps_exponential(np.maximum(cm, 1e-6), y))
            rm = np.concatenate(rm)
            rec.update(running_mean_exponential_crps=float(rm.mean()),
                       assumptions='No oracle: the per-sequence scale and memory of the correlated-exponential generator are '
                                   'not stored. Running mean = mean of the stored time_since_last_event values of events '
                                   '0..i-1 (including the first stored interval), floored at 1e-6, as an exponential mean.')
            crps = None
        if crps is not None:
            rec['oracle_gain_vs_marginal_percent'] = 100 * (rec['oracle_crps'] / rec['marginal_empirical_crps'] - 1)
            rec['per_decile'] = deciles(y_all, crps, marg)
        rec['seconds'] = round(time.time() - t0, 1)
        out[ds] = rec
        rows.append((ds, rec))
        print(ds, {k: v for k, v in rec.items() if not isinstance(v, (dict, str))}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print('\n%-20s %7s %10s %10s %10s %10s %8s' % ('dataset', 'n', 'oracle', 'marg.emp', 'marg.par', 'run.mean', 'gain%'))
    for ds, r in rows:
        f = lambda k: '%10.4f' % r[k] if k in r else '%10s' % '-'
        g = '%8.1f' % r['oracle_gain_vs_marginal_percent'] if 'oracle_gain_vs_marginal_percent' in r else '%8s' % '-'
        print('%-20s %7d %s %s %s %s %s' % (ds, r['n_targets'], f('oracle_crps'), f('marginal_empirical_crps'),
                                           f('marginal_parametric_crps'), f('running_mean_exponential_crps'), g))
    for ds, r in rows:
        if 'oracle_ll' in r:
            print('%-20s oracle LL %.4f   mass beyond grid (max) %.3g   quad check rel err %.2e' %
                  (ds, r['oracle_ll'], r['oracle_mass_beyond_grid_max'], r['oracle_quad_check_rel_err']))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
