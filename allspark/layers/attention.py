import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Computes scaled dot-product attention as defined in https://arxiv.org/pdf/1706.03762.pdf.

    Args:
        is_causal (bool, optional): If True, applies a causal mask to the attention weights.
        dropout (float, optional): Dropout probability applied to the attention weights.

    Scaled dot-product attention is defined as:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d))V
            where Q is the query tensor (shape: [batch_size, num_queries, d])
                  K is the key tensor (shape: [batch_size, num_keys, d])
                  V is the value tensor (shape: [batch_size, num_keys, d_v])

    Returns:
        torch.Tensor: The attention weighted sum of the value tensor (shape: [batch_size, num_queries, d_v])

    Note that d_v shoud be equal to d when chaining multiple attention layers.
    """

    def __init__(self, is_causal: bool = False, dropout: float = 0.0):
        super().__init__()
        self.is_causal = is_causal
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        d = key.shape[-1]
        weights = torch.matmul(query, key.transpose(-2, -1)) / (d**0.5)
        if self.is_causal:
            mask = torch.ones(1, *weights.shape[-2:]).tril_()
            weights.masked_fill_(mask == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, value)
