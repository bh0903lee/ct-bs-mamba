"""CT-BSMamba temporal point process model: MB-token embedding + temporal encoding,
a stack of CTBSMambaLayer blocks, optional LSTM, and IET / MB prediction heads."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.join(_HERE, 'thp_utils')
sys.path.insert(0, _UTILS)
import Constants

from ct_bsmamba_layer import CTBSMambaLayer


def get_non_pad_mask(seq: torch.Tensor) -> torch.Tensor:
    """(batch, seq_len) -> (batch, seq_len, 1) float mask, 0 on PAD."""
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).float().unsqueeze(-1)


class _TemporalEncoding(nn.Module):
    """Sinusoidal encoding of absolute event times."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        pos_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)]
        )
        self.register_buffer('pos_vec', pos_vec)

    def forward(self, time: torch.Tensor, non_pad_mask: torch.Tensor) -> torch.Tensor:
        """time: (B,L), returns (B,L,d_model)."""
        result = time.unsqueeze(-1) / self.pos_vec
        result = result.clone()
        result[..., 0::2] = torch.sin(result[..., 0::2])
        result[..., 1::2] = torch.cos(result[..., 1::2])
        return result * non_pad_mask


class CTBSMambaEncoder(nn.Module):
    """Stack of CTBSMambaLayer blocks over MB-token embedding + temporal encoding.

    mb_mode: 'rowmajor' (MB-token embedding, default) or 'bs_only' (no token embedding).
    """
    def __init__(
        self,
        num_types: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_layers: int,
        d_bm: int,
        use_ct: bool,
        use_bs: bool,
        ct_mode: str = 'log_ratio',
        dropout: float = 0.1,
        mb_mode: str = 'rowmajor',
    ):
        super().__init__()
        self.temporal_enc = _TemporalEncoding(d_model)
        self.mb_mode = mb_mode

        if mb_mode == 'rowmajor':
            # +1 for the PAD token (index 0)
            self.mb_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)
        elif mb_mode != 'bs_only':
            raise ValueError(f"Unknown mb_mode: {mb_mode}")

        self.layers = nn.ModuleList([
            CTBSMambaLayer(
                d_model=d_model, d_state=d_state, d_conv=d_conv,
                expand=expand, d_bm=d_bm, use_ct=use_ct,
                use_bs=use_bs, ct_mode=ct_mode,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        mb: torch.Tensor,          # (B, L)  MB tokens; also used for the PAD mask
        event_time: torch.Tensor,  # (B, L)
        iet: torch.Tensor,         # (B, L)
        B_t: torch.Tensor,         # (B, L)
        M_t: torch.Tensor,         # (B, L)
        non_pad_mask: torch.Tensor,# (B, L, 1)
    ) -> torch.Tensor:
        tem = self.temporal_enc(event_time, non_pad_mask)  # (B,L,d_model)

        if self.mb_mode == 'rowmajor':
            h = self.mb_emb(mb)                            # (B,L,d_model)
        else:  # bs_only
            h = torch.zeros_like(tem)

        for layer in self.layers:
            h = h + tem
            h = layer(h, iet=iet, B_t=B_t, M_t=M_t, non_pad_mask=non_pad_mask)
        return h


class _Predictor(nn.Module):
    def __init__(self, d_in: int, d_out: int, positivity: bool = False):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
        nn.init.xavier_normal_(self.linear.weight)
        # softplus output for the IET head (IET > 0)
        self.positivity = positivity

    def forward(self, x: torch.Tensor, non_pad_mask: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.positivity:
            out = F.softplus(out)
        return out * non_pad_mask


# Distributional IET head: K-component log-normal mixture p(tau | h), trained by
# next-IET NLL; the mixture mean serves as the point prediction.

class MixtureLogNormalHead(nn.Module):
    """Outputs raw mixture params of shape (..., 3K): [logits_w | mu | log_sigma]."""
    def __init__(self, d_in: int, n_mix: int = 8):
        super().__init__()
        self.n_mix = n_mix
        self.linear = nn.Linear(d_in, 3 * n_mix, bias=True)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, non_pad_mask: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)                 # (B, L, 3K)
        return out * non_pad_mask


def split_mixture(raw: torch.Tensor, K: int):
    """raw (..., 3K) -> (log_w, mu, log_sigma), each (..., K); log_sigma clamped to [-7, 1]."""
    log_w     = F.log_softmax(raw[..., :K], dim=-1)
    mu        = raw[..., K:2 * K]
    log_sigma = raw[..., 2 * K:3 * K].clamp(-7.0, 1.0)
    return log_w, mu, log_sigma


def mixture_nll(raw: torch.Tensor, tau: torch.Tensor, K: int,
                eps: float = 1e-6) -> torch.Tensor:
    """Per-element -log p(tau), p(tau) = sum_k w_k LogNormal(tau; mu_k, sigma_k).

    Returns a tensor with the shape of `tau`.
    """
    log_w, mu, log_sigma = split_mixture(raw, K)
    logtau = torch.log(tau.clamp(min=eps)).unsqueeze(-1)        # (..., 1)
    sigma  = log_sigma.exp()
    z      = (logtau - mu) / sigma
    comp_logpdf = (-0.5 * math.log(2 * math.pi) - log_sigma
                   - 0.5 * z * z - logtau)                       # (..., K)
    log_p = torch.logsumexp(log_w + comp_logpdf, dim=-1)         # (...)
    return -log_p


def mixture_mean(raw: torch.Tensor, K: int, max_val: float = 1e6) -> torch.Tensor:
    """Mixture mean E[tau] = sum_k w_k exp(mu_k + sigma_k^2 / 2), clamped at max_val."""
    log_w, mu, log_sigma = split_mixture(raw, K)
    sigma = log_sigma.exp()
    comp_mean = torch.exp((mu + 0.5 * sigma * sigma).clamp(max=math.log(max_val)))
    return (log_w.exp() * comp_mean).sum(-1)


class _RNNLayers(nn.Module):
    """Optional LSTM on top of the SSM encoder."""
    def __init__(self, d_model: int, d_rnn: int):
        super().__init__()
        self.rnn  = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.proj = nn.Linear(d_rnn, d_model)

    def forward(self, x: torch.Tensor, non_pad_mask: torch.Tensor) -> torch.Tensor:
        total_length = x.size(1)
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        lengths = lengths.clamp(min=1)
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _  = self.rnn(packed)
        out, _  = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=total_length
        )
        return self.proj(out)


class CTBSMamba(nn.Module):
    """CT-BSMamba model. use_ct / use_bs toggle the CT and BS terms; both False gives a
    plain Mamba encoder. iet_head: 'point' (softplus) or 'mixture' (log-normal mixture)."""

    def __init__(
        self,
        num_types: int,
        d_model: int   = 128,
        d_rnn: int     = 64,
        d_state: int   = 16,
        d_conv: int    = 4,
        expand: int    = 2,
        n_layers: int  = 4,
        d_bm: int      = 8,
        use_ct: bool   = True,
        use_bs: bool   = True,
        ct_mode: str   = 'log_ratio',
        dropout: float = 0.1,
        mb_mode: str   = 'rowmajor',
        iet_head: str  = 'point',
        n_mix: int     = 8,
        no_rnn: bool   = False,
    ):
        super().__init__()
        self.num_types = num_types
        self.mb_mode   = mb_mode
        self.iet_head  = iet_head
        self.n_mix     = n_mix
        self.no_rnn    = no_rnn

        self.encoder = CTBSMambaEncoder(
            num_types=num_types, d_model=d_model,
            d_state=d_state, d_conv=d_conv, expand=expand,
            n_layers=n_layers, d_bm=d_bm,
            use_ct=use_ct, use_bs=use_bs,
            ct_mode=ct_mode, dropout=dropout,
            mb_mode=mb_mode,
        )
        # --no_rnn drops the LSTM (pure-Mamba backbone)
        self.rnn = None if no_rnn else _RNNLayers(d_model, d_rnn)

        # Hawkes intensity parameters used by Utils.log_likelihood
        self.linear = nn.Linear(d_model, num_types)
        self.alpha  = nn.Parameter(torch.tensor(-0.1))
        self.beta   = nn.Parameter(torch.tensor(1.0))

        if iet_head == 'mixture':
            self.time_predictor = MixtureLogNormalHead(d_model, n_mix=n_mix)
        else:
            self.time_predictor = _Predictor(d_model, 1, positivity=True)
        self.mb_predictor   = _Predictor(d_model, num_types)

    def forward(
        self,
        mb: torch.Tensor,
        event_time: torch.Tensor,
        iet: torch.Tensor,
        B_t: torch.Tensor,
        M_t: torch.Tensor,
    ):
        """Returns (enc_out (B, L, d_model), time_pred (B, L, 1 or 3K), mb_pred (B, L, num_types))."""
        non_pad_mask = get_non_pad_mask(mb)                 # (B,L,1)
        enc = self.encoder(mb, event_time, iet, B_t, M_t, non_pad_mask)
        if self.no_rnn:
            enc = enc * non_pad_mask
        else:
            enc = self.rnn(enc, non_pad_mask) * non_pad_mask
        t_pred  = self.time_predictor(enc, non_pad_mask)
        mb_pred = self.mb_predictor(enc, non_pad_mask)      # (B,L,num_types)
        return enc, t_pred, mb_pred
