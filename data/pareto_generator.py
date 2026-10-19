"""Fixed benchmark generator; numerical operations preserved from the evaluated implementation."""
import numpy as np
from scipy.stats import norm as _norm
def gen_corr_pareto_sequence(n_events: int, alpha: float, x_min: float,
                              rho: float, rng: np.random.Generator) -> list:
    """Generate one sequence of correlated Pareto IETs."""
    # AR(1) Gaussian process
    z = np.zeros(n_events)
    z[0] = rng.standard_normal()
    eps = rng.standard_normal(n_events)
    for t in range(1, n_events):
        z[t] = rho * z[t - 1] + np.sqrt(1 - rho ** 2) * eps[t]

    # Copula transform: Gaussian → Uniform → Pareto
    u = _norm.cdf(z)
    # Clip away from 0/1 to avoid inf
    u = np.clip(u, 1e-9, 1 - 1e-9)
    tau = x_min * (1 - u) ** (-1.0 / alpha)

    # Build event sequence
    seq = []
    t_cum = 0.0
    for i, iet in enumerate(tau):
        t_cum += float(iet)
        seq.append({
            'time_since_start':      round(t_cum, 6),
            'time_since_last_event': round(float(iet), 6),
            'type_event':            0,   # single type (Dataset.py does +1)
        })
    return seq

def gen_dataset(n_train: int, n_dev: int, n_test: int,
                seq_len_mu: int, seq_len_sig: int,
                alpha: float, x_min: float, rho: float,
                seed: int = 42) -> dict:
    """Generate train/dev/test splits."""
    rng = np.random.default_rng(seed)

    def make_split(n):
        data = {}
        for i in range(n):
            L = max(10, int(rng.normal(seq_len_mu, seq_len_sig)))
            data[i] = gen_corr_pareto_sequence(L, alpha, x_min, rho, rng)
        return data

    return {
        'train': make_split(n_train),
        'dev':   make_split(n_dev),
        'test':  make_split(n_test),
        'dim_process': 1,
    }
