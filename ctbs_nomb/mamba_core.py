"""Pure-PyTorch Mamba SSM block whose forward() takes an optional `delta_bias`
added to the raw Δ before softplus (the hook used by the CT and BS terms)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(A_bar: torch.Tensor, Bx: torch.Tensor) -> torch.Tensor:
    """Sequential recurrence h_t = A_bar_t * h_{t-1} + Bx_t, h_0 = 0.

    A_bar, Bx: (batch, seq_len, d_state). Returns h of the same shape.
    """
    batch, seq_len, d_state = Bx.shape
    h = torch.zeros(batch, d_state, device=Bx.device, dtype=Bx.dtype)
    outs = []
    for t in range(seq_len):
        h = A_bar[:, t, :] * h + Bx[:, t, :]
        outs.append(h.unsqueeze(1))
    return torch.cat(outs, dim=1)


class MambaSSM(nn.Module):
    """Pre-norm Mamba residual block (d_inner = expand * d_model).

    `delta_bias` (batch, seq_len) is added to the raw Δ projection before softplus.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv  = d_conv

        self.norm = nn.LayerNorm(d_model)

        # x → [x_inner, z], both of width d_inner
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # causal depthwise conv; output is cropped to L in forward()
        self.conv1d = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            groups       = self.d_inner,
            padding      = d_conv - 1,
            bias         = True,
        )

        # conv_out → [B_ssm (d_state), C_ssm (d_state), dt_raw (1)]
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + 1, bias=False)

        # scalar dt_raw -> d_inner timescale projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A = -exp(A_log), initialised to -(1..d_state) per channel
        A_log = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
        ).unsqueeze(0).expand(self.d_inner, -1).clone()   # (d_inner, d_state)
        self.A_log = nn.Parameter(A_log)

        # per-channel skip weight
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop     = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # initial Δ = softplus(bias) is log-uniform in [0.001, 0.1]
        dt_init_std = self.d_inner ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_bias = torch.exp(
            torch.rand(self.d_inner) * (torch.log(torch.tensor(0.1)) -
                                        torch.log(torch.tensor(0.001)))
            + torch.log(torch.tensor(0.001))
        )
        inv_softplus = torch.log(torch.expm1(dt_bias))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus)

    def forward(
        self,
        x: torch.Tensor,
        delta_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """x: (batch, seq_len, d_model); delta_bias: (batch, seq_len) or None.

        A positive delta_bias gives a larger Δ, i.e. faster forgetting.
        """
        batch, seq_len, _ = x.shape
        residual = x

        x = self.norm(x)

        xz       = self.in_proj(x)                      # (B, L, 2·d_inner)
        x_inner, z = xz.chunk(2, dim=-1)                # each (B, L, d_inner)

        # causal conv over time: crop the right padding back to L
        x_conv = (
            self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len]
                 .transpose(1, 2)
        )                                                # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        params = self.x_proj(x_conv)                    # (B, L, 2·d_state+1)
        B_ssm  = params[..., : self.d_state]            # (B, L, d_state)
        C_ssm  = params[..., self.d_state: 2*self.d_state]  # (B, L, d_state)
        dt_raw = params[..., -1:]                       # (B, L, 1)

        dt = self.dt_proj(dt_raw)                       # (B, L, d_inner)
        if delta_bias is not None:
            # broadcast over d_inner
            dt = dt + delta_bias.unsqueeze(-1)
        dt = F.softplus(dt)                             # (B, L, d_inner), > 0

        # discretise: A_bar = exp(Δ A), B_bar x = Δ B x
        A = -torch.exp(self.A_log.float())              # (d_inner, d_state)

        delta_A = torch.einsum('bln,ns->blns', dt, A)  # (B,L,d_inner,d_state)
        A_bar   = torch.exp(delta_A)                   # (B,L,d_inner,d_state)

        Bx = torch.einsum('bln,bls->blns', dt * x_conv, B_ssm)  # (B,L,d_inner,d_state)

        # fold d_inner into the batch dimension for the scan
        BN = batch * self.d_inner
        A_bar_flat = A_bar.permute(0, 2, 1, 3).reshape(BN, seq_len, self.d_state)
        Bx_flat    = Bx.permute(0, 2, 1, 3).reshape(BN, seq_len, self.d_state)

        h_flat = selective_scan(A_bar_flat, Bx_flat)   # (B·d_inner, L, d_state)

        h = (
            h_flat
            .reshape(batch, self.d_inner, seq_len, self.d_state)
            .permute(0, 2, 1, 3)
        )

        # y = C h + D x_conv, gated by SiLU(z)
        y = torch.einsum('blns,bls->bln', h, C_ssm)    # (B, L, d_inner)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        y = y * F.silu(z)                               # (B, L, d_inner)

        return self.drop(self.out_proj(y)) + residual   # (B, L, d_model)
