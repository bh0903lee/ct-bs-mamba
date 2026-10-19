"""CT-BSMamba layer: a Mamba block whose step size Δ is shifted by a continuous-time (CT)
term and by a burstiness/memory term g([B_t, M_t]), with an additive [B_t, M_t] output map.

CT term of the reported model (CTBS_CT_POLES=R > 1): a bank of R causal EMA references
mu_t^(r) of the past inter-event times, with windows log-spaced in [2, CTBS_CT_WMAX], and
learned per-reference weights alpha_r; the Δ bias is sum_r alpha_r * log(tau_t / mu_t^(r)).
With CTBS_CT_POLES=1 (default) a single EMA reference (CTBS_EMA_BETA, CTBS_EMA_K) is used.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_core import MambaSSM

# Fixed (non-learnable) EMA kernels, set from the environment at import time.
import os as _os
_EMA_BETA = float(_os.environ.get('CTBS_EMA_BETA', 0.9))
_EMA_K    = int(_os.environ.get('CTBS_EMA_K', 32))   # single-EMA kernel length
_CT_POLES = int(_os.environ.get('CTBS_CT_POLES', 1))  # number of EMA references R
_CT_WMAX  = float(_os.environ.get('CTBS_CT_WMAX', 64.0))  # longest EMA window


class CTBSMambaLayer(nn.Module):
    """Residual Mamba block with optional CT (use_ct) and BS (use_bs) Δ biases.

    ct_mode: 'log_ratio' (alpha * log(tau / mu), the reported model) or
             'log_abs' (alpha * log(tau), no EMA reference).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        d_bm: int = 8,
        use_ct: bool = True,
        use_bs: bool = True,
        ct_mode: str = 'log_ratio',
        dropout: float = 0.1,
    ):
        super().__init__()
        if ct_mode not in ('log_ratio', 'log_abs'):
            raise ValueError(f"Unknown ct_mode: {ct_mode}")
        self.use_ct  = use_ct
        self.use_bs  = use_bs
        self.ct_mode = ct_mode

        self.mamba = MambaSSM(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand, dropout=dropout,
        )

        if use_ct:
            if _CT_POLES > 1 and ct_mode == 'log_ratio':
                # reference bank: R EMA references with windows log-spaced in [2, _CT_WMAX],
                # one learnable weight per reference
                Kp = int(max(128, round(5 * _CT_WMAX)))        # kernel long enough for the longest EMA
                wins = torch.exp(torch.linspace(math.log(2.0), math.log(_CT_WMAX), _CT_POLES))
                betas = 1.0 - 1.0 / wins
                kern = torch.stack([
                    (1 - b) * b ** torch.arange(Kp - 1, -1, -1, dtype=torch.float32)
                    for b in betas
                ])                                                 # (R, K)
                self.register_buffer('_ema_kernel', kern.view(_CT_POLES, 1, Kp))
                self.alpha = nn.Parameter(torch.full((_CT_POLES,), 0.1 / _CT_POLES))
            else:
                self.alpha = nn.Parameter(torch.tensor(0.1))
                if ct_mode == 'log_ratio':
                    # causal geometric-decay EMA kernel
                    weights = (1 - _EMA_BETA) * _EMA_BETA ** torch.arange(_EMA_K - 1, -1, -1, dtype=torch.float32)
                    self.register_buffer('_ema_kernel', weights.view(1, 1, _EMA_K))

        if use_bs:
            # [B, M] -> scalar Δ bias
            self.bm_delta = nn.Sequential(
                nn.Linear(2, d_bm), nn.SiLU(), nn.Linear(d_bm, 1)
            )
            # [B, M] -> d_model additive output term
            self.bm_out = nn.Sequential(
                nn.Linear(2, d_bm), nn.SiLU(), nn.Linear(d_bm, d_model)
            )

    def _ct_bias(self, iet: torch.Tensor) -> torch.Tensor:
        """CT Δ bias from inter-event times iet (B, L); returns (B, L)."""
        eps = 1e-6
        tau = iet.clamp(min=eps)

        if self.ct_mode == 'log_abs':
            return self.alpha * torch.log(tau)

        # EMA references mu_t over past IETs (left-padded with eps)
        K = self._ema_kernel.shape[-1]
        R = self._ema_kernel.shape[0]                          # 1 = single EMA, >1 = reference bank
        padded = F.pad(tau.unsqueeze(1), (K, 0), value=eps)   # (B, 1, L+K)
        mu = F.conv1d(padded, self._ema_kernel)[..., :tau.shape[1]]  # (B, R, L)
        mu = mu.clamp(min=eps)
        log_ratio = torch.log(tau.unsqueeze(1) / mu)          # (B, R, L)

        if R > 1:
            return (self.alpha.view(1, R, 1) * log_ratio).sum(1)   # (B, L) reference bank
        return self.alpha * log_ratio[:, 0, :]                     # (B, L) single EMA

    def _bs_delta_bias(self, bm: torch.Tensor) -> torch.Tensor:
        """BS Δ bias from bm = [B_t, M_t] (batch, seq_len, 2); returns (batch, seq_len)."""
        return self.bm_delta(bm).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        iet: torch.Tensor,
        B_t: torch.Tensor,
        M_t: torch.Tensor,
        non_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """x: (B, L, d_model); iet, B_t, M_t: (B, L); non_pad_mask: (B, L, 1), 0 on padding.

        Returns (B, L, d_model).
        """
        delta_bias = None

        if self.use_ct:
            delta_bias = self._ct_bias(iet)                 # (B, L)

        bm = torch.stack([B_t, M_t], dim=-1)               # (B, L, 2)
        if self.use_bs:
            bs_bias = self._bs_delta_bias(bm)               # (B, L)
            delta_bias = bs_bias if delta_bias is None else delta_bias + bs_bias

        out = self.mamba(x, delta_bias=delta_bias)          # (B, L, d_model)

        if self.use_bs:
            out = out + self.bm_out(bm)                     # (B, L, d_model)

        if non_pad_mask is not None:
            out = out * non_pad_mask

        return out
