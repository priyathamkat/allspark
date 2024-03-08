import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()

        positions = torch.arange(max_seq_len).unsqueeze(-1)

        dimensions = 2 * torch.arange((d_model + 1) // 2).unsqueeze(0)
        frequencies = 1 / (10000 ** (dimensions / d_model))
        
        arguments = positions * frequencies

        positional_encoding = torch.zeros(max_seq_len, d_model)
        positional_encoding[:, ::2] = torch.sin(arguments)
        positional_encoding[:, 1::2] = torch.cos(arguments[:, : d_model // 2])

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_encoding[: x.shape[-2], :].to(x)
