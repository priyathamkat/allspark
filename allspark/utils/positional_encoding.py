import torch


def add_positional_encoding(x: torch.Tensor) -> torch.Tensor:
    """Adds positional encoding to the input tensor.

    Positional encoding is defined as:
        PE(pos, 2i) = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        where pos is the position and i is the dimension

    Args:
        x (torch.Tensor): Input tensor (shape: [batch_size, sequence_length, d])

    Returns:
        torch.Tensor: Input tensor with positional encoding added (shape: [batch_size, sequence_length, d])
    """
    d = x.shape[-1]

    positions = torch.arange(x.shape[-2]).unsqueeze(-1)

    dimensions = 2 * torch.arange((d + 1) // 2)
    dimensions = dimensions.unsqueeze(0)

    frequencies = 1 / (10000 ** (dimensions / d))
    arguments = positions * frequencies

    sinusoids = torch.exp(torch.complex(torch.tensor(0.0), arguments))
    sines = torch.imag(sinusoids)
    cosines = torch.real(sinusoids)

    positional_encoding = torch.zeros_like(x)
    positional_encoding[:, :, ::2] = sines
    positional_encoding[:, :, 1::2] = cosines[:, : d // 2]

    return x + positional_encoding
