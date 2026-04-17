import torch
import torch.nn as nn

class ChannelLayer(nn.Module):
    """
    Differentiable AWGN channel layer.

    Sits between encoder and decoder in the compute graph.
    Adds noise during BOTH training and evaluation —
    because we are always operating over a real channel.

    Gradients flow through this layer to the encoder
    because noise is additive: d(z + noise)/dz = 1.

    Call set_snr() to simulate channel mismatch at test time.
    """

    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db     = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)

    def set_snr(self, snr_db: float):
        """Swap SNR at test time to simulate channel mismatch."""
        self.snr_db     = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z     : encoder output (batch, bottleneck_dim)
        return: noisy received signal, same shape
        """
        signal_power = z.pow(2).mean().detach()
        noise_std    = torch.sqrt(signal_power / self.snr_linear)
        return z + noise_std * torch.randn_like(z)