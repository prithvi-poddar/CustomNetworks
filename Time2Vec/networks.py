import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

        # Linear term
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))

        # Periodic terms
        self.W = nn.Parameter(torch.randn(out_features - 1))
        self.B = nn.Parameter(torch.zeros(out_features - 1))

    def forward(self, t):
        # t shape: (batch_size, 1) or (batch_size,)
        t = t.unsqueeze(-1) if t.dim() == 1 else t

        linear = self.w0 * t + self.b0
        periodic = torch.sin(t * self.W + self.B)

        return torch.cat([linear, periodic], dim=-1)
