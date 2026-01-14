import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_drop: float = 0.1,
        out_drop: float = 0.1,
        bias: bool = True,
        attn_score_only: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.nh = num_heads
        self.head_dim = hidden_size // num_heads

        self.Wq = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.Wkv = nn.Linear(hidden_size, hidden_size * 2, bias=bias)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)
        self.attn_score_only = attn_score_only

    def forward(self, query, kv, mask=None):
        """
        query: (B, 1, C)     ← time / agent state
        kv:    (B, N, C)     ← task nodes
        mask:  (B, N)        ← valid tasks
        """
        B, Q_len, C = query.shape
        _, K_len, _ = kv.shape

        # Project
        q = self.Wq(query)  # (B, 1, C)
        kv = self.Wkv(kv)  # (B, N, 2C)

        # Reshape
        q = q.view(B, Q_len, self.nh, self.head_dim).transpose(1, 2)
        kv = kv.view(B, K_len, 2, self.nh, self.head_dim).permute(0, 3, 2, 1, 4)
        k, v = kv.unbind(dim=2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            mask = ~mask.bool()
            attn = attn.masked_fill(mask[:, None, None, :], -1e9)

        if self.attn_score_only:
            attn = attn.squeeze(2)
            attn = attn.sum(dim=1) / np.sqrt(self.nh)
            return attn

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate
        out = attn @ v  # (B, nh, 1, head_dim)
        out = out.transpose(1, 2).reshape(B, Q_len, C)

        return self.out_drop(self.Wo(out)).squeeze(1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_drop: float = 0.1,
        out_drop: float = 0.1,
        bias: bool = True,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_heads == 0
        self.nh = num_heads
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x, mask):
        B, S, C = x.shape
        mask = ~mask.to(torch.bool)

        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C // self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)

        attn = attn / np.sqrt(k.size(-1))
        attn = attn.masked_fill(mask.view(B, 1, 1, S), float(-100000))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v

        x = x.transpose(1, 2).reshape(B, S, C)

        return self.out_drop(self.Wo(x))
