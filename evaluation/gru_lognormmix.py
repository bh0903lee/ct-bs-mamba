"""GRU-LogNormMix reference (intensity-free model of Shchur et al., 2020): a GRU over log(IET)
with the same K-component log-normal mixture head as CT-BS-Mamba and no burstiness inputs.
This module holds the model, data loading and the validation evaluator; the reported models
are trained and scored with evaluation/train_gru.py.
"""
import math, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path("datasets")      # {ds}/{train,dev,test}.pkl; train_gru.py sets it from --data_dir


# Log-normal mixture head (same as CT-BS-Mamba).
def split_mixture(raw, K):
    log_w = F.log_softmax(raw[..., :K], dim=-1)
    mu = raw[..., K:2*K]
    log_sigma = raw[..., 2*K:3*K].clamp(-7.0, 1.0)
    return log_w, mu, log_sigma

def mixture_nll(raw, tau, K, eps=1e-6):
    log_w, mu, log_sigma = split_mixture(raw, K)
    logtau = torch.log(tau.clamp(min=eps)).unsqueeze(-1)
    sigma = log_sigma.exp(); z = (logtau - mu)/sigma
    comp = -0.5*math.log(2*math.pi) - log_sigma - 0.5*z*z - logtau
    return -torch.logsumexp(log_w + comp, dim=-1)

def mixture_mean(raw, K, max_val=1e6):
    log_w, mu, log_sigma = split_mixture(raw, K)
    sigma = log_sigma.exp()
    cm = torch.exp((mu + 0.5*sigma*sigma).clamp(max=math.log(max_val)))
    return (log_w.exp()*cm).sum(-1)

def mixture_sample(raw, K, M, max_val=1e6):
    """Draw M samples per position: (..., M)."""
    log_w, mu, log_sigma = split_mixture(raw, K)
    sigma = log_sigma.exp()
    w = log_w.exp()                                  # (..., K)
    idx = torch.multinomial(w.reshape(-1, K), M, replacement=True).reshape(*w.shape[:-1], M)
    mu_s = torch.gather(mu, -1, idx); sig_s = torch.gather(sigma, -1, idx)
    z = torch.randn_like(mu_s)
    return torch.exp((mu_s + sig_s*z).clamp(max=math.log(max_val)))


class LogNormMix(nn.Module):
    def __init__(self, d_model=256, K=8, n_layers=1):
        super().__init__()
        self.K = K
        self.inp = nn.Linear(1, d_model)             # embed log(IET)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, 3*K)
        nn.init.xavier_normal_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, logiet):                        # logiet (B, L)
        x = self.inp(logiet.unsqueeze(-1))            # (B, L, d)
        h, _ = self.gru(x)                            # (B, L, d)
        return self.head(h)                           # (B, L, 3K) params for NEXT iet


def load_seqs(ds, split):
    p = DATA / ds / f"{split}.pkl"
    d = pickle.load(open(p, "rb"), encoding="latin-1")
    if split not in d:
        raise KeyError(f"{p} has no '{split}' key (found {sorted(k for k in d if k != 'dim_process')})")
    data = d[split]
    # inner may be a list-of-sequences OR a dict {seq_id: sequence}
    iters = data.values() if isinstance(data, dict) else data
    seqs = []
    for s in iters:
        iet = np.array([e["time_since_last_event"] for e in s], dtype=np.float64)
        if len(iet) >= 3:
            seqs.append(iet)
    return seqs

def batches(seqs, bs, device, shuffle=True):
    order = np.random.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for i in range(0, len(seqs), bs):
        chunk = [seqs[j] for j in order[i:i+bs]]
        L = max(len(s) for s in chunk)
        iet = np.zeros((len(chunk), L)); mask = np.zeros((len(chunk), L))
        for b, s in enumerate(chunk):
            iet[b, :len(s)] = s; mask[b, :len(s)] = 1
        yield (torch.tensor(iet, dtype=torch.float32, device=device),
               torch.tensor(mask, dtype=torch.float32, device=device))

def energy_crps_np(samples, y):
    """Unbiased (U-statistic) energy-form CRPS; samples (N, M), y (N,)."""
    M = samples.shape[1]
    if M < 2:
        raise ValueError(f'the U-statistic needs at least 2 samples, got {M}')
    t1 = np.abs(samples - y[:, None]).mean(1)
    xs = np.sort(samples, 1); coef = 2*np.arange(1, M+1) - M - 1
    t2 = (2.0/(M*(M-1)))*(coef[None, :]*xs).sum(1)      # = 2/(M(M-1)) * sum_{i<j}|Xi-Xj|
    return t1 - 0.5*t2


def evaluate(model, seqs, device, K, M=100):
    model.eval(); nll_sum=0.0; se_sum=0.0; n=0; crps_parts=[]
    with torch.no_grad():
        for iet, mask in batches(seqs, 32, device, shuffle=False):
            logiet = torch.log(iet.clamp(min=1e-6))
            raw = model(logiet)                       # predict tau_{i+1} from history<=i
            # params at position i predict iet[:, i+1]
            raw = raw[:, :-1, :]; tgt = iet[:, 1:]; m = (mask[:, 1:]*mask[:, :-1])
            nll = mixture_nll(raw, tgt, K) * m
            mean = mixture_mean(raw, K)
            se = ((mean - tgt)**2) * m
            nll_sum += nll.sum().item(); se_sum += se.sum().item(); n += m.sum().item()
            # CRPS over valid positions
            samp = mixture_sample(raw, K, M)          # (B, L-1, M)
            mm = m.bool().cpu().numpy()
            s = samp.cpu().numpy()[mm]; yv = tgt.cpu().numpy()[mm]
            if len(yv): crps_parts.append(energy_crps_np(s, yv))
    ll = -nll_sum/max(n,1); rmse = math.sqrt(se_sum/max(n,1))
    crps = float(np.concatenate(crps_parts).mean()) if crps_parts else float("nan")
    return ll, rmse, crps
