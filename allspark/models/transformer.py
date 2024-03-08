import torch
import torch.nn as nn

from allspark.layers.attention import MultiHeadAttention
from allspark.layers.ffn import PositionWiseFeedForwardNet
from allspark.layers.positional_encoding import PositionalEncoding


class TransformerEncoderStack(nn.Module):
    def __init__(
        self, num_heads: int, d_model: int, d_k: int, d_v: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout
        )
        self.ffn = PositionWiseFeedForwardNet(d_model)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

    def forward(self, x: torch.Tensor):
        x = self.layer_norms[0](x + self.self_attention(x, x, x))
        x = self.layer_norms[1](x + self.ffn(x))
        return x


class TransformerDecoderStack(nn.Module):
    def __init__(
        self, num_heads: int, d_model: int, d_k: int, d_v: int, dropout: float
    ) -> None:
        super().__init__()
        self.encoder_decoder_attention = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout
        )
        self.masked_self_attention = MultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            is_causal=True,
            dropout=dropout,
        )
        self.ffn = PositionWiseFeedForwardNet(d_model)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        x = self.layer_norms[0](x + self.masked_self_attention(x, x, x))
        x = self.layer_norms[1](
            x + self.encoder_decoder_attention(x, encoder_output, encoder_output)
        )
        x = self.layer_norms[2](x + self.ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        vocabulary_size: int,
        num_stacks: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.source_embedding = nn.Embedding(vocabulary_size, d_model)
        self.target_embedding = nn.Embedding(vocabulary_size, d_model)
        self.encoder_stacks = nn.ModuleList(
            [
                TransformerEncoderStack(num_heads, d_model, d_k, d_v, dropout)
                for _ in range(num_stacks)
            ]
        )
        self.decoder_stacks = nn.ModuleList(
            [
                TransformerDecoderStack(num_heads, d_model, d_k, d_v, dropout)
                for _ in range(num_stacks)
            ]
        )
        self.linear = nn.Linear(d_model, vocabulary_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        source = self.source_embedding(source)
        source_encoding = self.positional_encoding(source)
        for stack in self.encoder_stacks:
            source_encoding = stack(source_encoding)

        target = self.target_embedding(target)
        target_encoding = self.positional_encoding(target)
        for stack in self.decoder_stacks:
            target_encoding = stack(target_encoding, source_encoding)

        output = self.linear(target_encoding)
        output = self.softmax(output)
        return output
