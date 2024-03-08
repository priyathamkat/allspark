import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Computes scaled dot-product attention as defined in https://arxiv.org/pdf/1706.03762.pdf.

    This layer has no learnable parameters.

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

    def __init__(self, is_causal: bool = False, dropout: float = 0.0) -> None:
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
            mask = torch.ones(*weights.shape[-2:]).tril_()
            weights.masked_fill_(mask == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, value)


def init_xavier_normal_parameter(shape: torch.Size) -> nn.Parameter:
    data = torch.empty(shape)
    nn.init.xavier_normal_(data)
    return nn.Parameter(data)


class MultiHeadAttention(nn.Module):
    """Computes multi-head attention as defined in https://arxiv.org/pdf/1706.03762.pdf.

    Args:
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of model embeddings.
        d_k (int): Dimensionality of the key tensor.
        d_v (int): Dimensionality of the value tensor.
        is_causal (bool, optional): If True, applies a causal mask to the attention weights.
        dropout (float, optional): Dropout probability applied to the attention weights.

    Multi-head attention is defined as:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W_o
            where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
                  W^Q_i, W^K_i, W^V_i, W_o are learnable parameters
                  W_o is a learnable parameter that projects the concatenated attention heads to the output space

    Returns:
        torch.Tensor: The multi-head attention tensor (shape: [batch_size, num_queries, d_model])

    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        is_causal: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_q = init_xavier_normal_parameter((num_heads, d_model, d_k))
        self.w_k = init_xavier_normal_parameter((num_heads, d_model, d_k))
        self.w_v = init_xavier_normal_parameter((num_heads, d_model, d_v))
        self.w_o = init_xavier_normal_parameter((d_model, num_heads * d_v))
        self.attention = ScaledDotProductAttention(is_causal, dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        query = torch.matmul(query.unsqueeze(1), self.w_q)
        key = torch.matmul(key.unsqueeze(1), self.w_k)
        value = torch.matmul(value.unsqueeze(1), self.w_v)
        output = self.attention(query, key, value)
        output = output.reshape((output.shape[0], -1, output.shape[-1]))
        output = torch.matmul(self.w_o, output)
        return output
