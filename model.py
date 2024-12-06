import math
import torch
from torch import nn
from torch.nn import functional as F


class FourierEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lo: float = 1.0,
        hi: float = 10.0,
    ) -> None:
        super().__init__()
        cnt = out_channels // (in_channels * 2)
        ll, lh = math.log(lo), math.log(hi)
        self.factors = nn.Parameter(
            torch.exp(torch.linspace(ll, lh, cnt)), requires_grad=False
        )

    def forward(self, x):
        # x: (..., in_channels)
        # factors: (cnt, )
        # y: (..., in_channels, cnt)
        y = x.unsqueeze(-1) * self.factors
        y = torch.cat([torch.cos(y), torch.sin(y)], dim=-1)
        return y.flatten(-2)


class Stan(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(in_channels))

    def reset_parameters(self):
        nn.init.normal_(self.weights, mean=1, std=0.1)

    def forward(self, x):
        return F.tanh(x) + x * F.tanh(x) * self.weights


class PINN(nn.Module):
    def __init__(
        self, in_channels, hid_channels, out_channels, num_hiddens, scaling=1.0
    ):
        super().__init__()
        self.encoder = FourierEmbedding(
            in_channels, hid_channels, 1 / scaling, 20 / scaling
        )

        self.models = nn.ModuleList()
        for _ in range(num_hiddens - 1):
            self.models.append(
                nn.Sequential(
                    nn.Linear(hid_channels, hid_channels),
                    Stan(hid_channels),
                )
            )
        self.decoder = nn.Linear(hid_channels, out_channels)

        self.scaling = scaling

    def muon_params(self):
        return [p for p in self.models.parameters() if p.ndim >= 2]

    def adamw_params(self):
        params = [p for p in self.models.parameters() if p.ndim < 2]
        params += list(self.decoder.parameters())
        return params

    def forward(self, x):
        y = self.encoder(x * self.scaling)
        for model in self.models:
            y = model(y)
        return self.decoder(y)

    def reset_parameters(self):
        self.models.apply(lambda m: getattr(m, "reset_parameters", lambda: None)())
        self.decoder.apply(lambda m: getattr(m, "reset_parameters", lambda: None)())
