"""Fixed benchmark generator; numerical operations preserved from the evaluated implementation."""
import numpy as np
from collections import deque
def lomax_sample(gamma, beta, size=1):
    """
    Sample from Lomax distribution (Pareto Type II):
    f(τ) = (γ-1)·β · (1+β·τ)^{-γ},  τ ≥ 0
    Inverse CDF: τ = ((1-U)^{-1/(γ-1)} - 1) / β
    """
    u = np.random.uniform(size=size)
    return ((1 - u) ** (-1.0 / (gamma - 1)) - 1.0) / beta

def simulate_powerlaw_hawkes_cluster(
    mu: float,
    alpha: float,
    beta: float,
    gamma: float,
    T_max: float,
    seed: int = None,
) -> list:
    """
    Simulate one realization of Hawkes process with Lomax kernel via the
    cluster (Galton-Watson branching) representation.

    Process: λ(t|H) = μ + Σ_{t_i<t} α(γ-1)β(1+β(t-t_i))^{-γ}

    Args:
        mu    : baseline (immigrant) Poisson rate
        alpha : branching ratio ∈ (0,1) — mean offspring per event
        beta  : time-scale parameter of Lomax kernel
        gamma : power-law decay exponent (> 1 for finite integral)
        T_max : observation window [0, T_max]

    Returns:
        Sorted list of event times in [0, T_max].
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: immigrants from Poisson(mu)
    n_immigrants = np.random.poisson(mu * T_max)
    immigrants   = np.sort(np.random.uniform(0, T_max, n_immigrants))

    all_events = list(immigrants)
    queue      = deque(immigrants)   # events waiting to generate offspring

    # Step 2: branching process
    # Each event generates N_children ~ Poisson(alpha).
    # Each child at delay Δ ~ Lomax(gamma, beta) from parent.
    # Recurse until queue empty (terminates a.s. because alpha < 1).
    MAX_EVENTS = 200_000   # safety cap
    while queue and len(all_events) < MAX_EVENTS:
        t_parent = queue.popleft()

        # Number of direct children
        n_children = np.random.poisson(alpha)
        if n_children == 0:
            continue

        # Children delays from Lomax
        delays   = lomax_sample(gamma, beta, size=n_children)
        children = t_parent + delays

        for t_child in children:
            if t_child < T_max:
                all_events.append(t_child)
                queue.append(t_child)

    return sorted(all_events)

def generate_dataset(
    n_train: int = 3000,
    n_valid: int = 400,
    n_test:  int = 600,
    mu:    float = 0.3,
    alpha: float = 0.7,
    beta:  float = 1.0,
    gamma: float = 2.2,
    T_max: float = 50.0,
    min_len: int = 5,
    max_len: int = 500,
    seed: int = 42,
    name: str = "syn_powerlaw_01",
):
    """
    Generate train/valid/test splits in THP-compatible pickle format.

    Returns dict with dim_process=1 and lists of sequences.
    Each sequence = list of {'time_since_start': t, 'type_event': 1}
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    total = n_train + n_valid + n_test
    print(f"Simulating {total} sequences (mu={mu}, alpha={alpha}, beta={beta}, gamma={gamma}, T_max={T_max})")

    sequences = []
    attempts  = 0
    max_att   = total * 20

    while len(sequences) < total and attempts < max_att:
        attempts += 1
        seed_i = seed + attempts
        evts = simulate_powerlaw_hawkes_cluster(mu, alpha, beta, gamma, T_max, seed=seed_i)
        if min_len <= len(evts) <= max_len:
            sequences.append(evts)

    if len(sequences) < total:
        raise RuntimeError(f"Only {len(sequences)}/{total} sequences pass length filter "
                           f"({min_len}–{max_len}). Try adjusting parameters.")

    np.random.shuffle(sequences)
    train_seqs = sequences[:n_train]
    valid_seqs = sequences[n_train:n_train+n_valid]
    test_seqs  = sequences[n_train+n_valid:]

    def to_thp_format(seqs):
        result = []
        for evts in seqs:
            if len(evts) < 2:
                continue
            t0 = evts[0]
            result.append([{'time_since_start': t - t0, 'type_event': 1}
                            for t in evts])
        return result

    data = {
        'train':       to_thp_format(train_seqs),
        'dev':         to_thp_format(valid_seqs),
        'test':        to_thp_format(test_seqs),
        'dim_process': 1,
        'args':        {'mu': mu, 'alpha': alpha, 'beta': beta, 'gamma': gamma,
                        'T_max': T_max, 'name': name},
    }
    np.random.set_state(rng_state)
    print(f"Generated: train={len(data['train'])}, dev={len(data['dev'])}, test={len(data['test'])}")
    return data
