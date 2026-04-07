import torch
import torch.nn as nn

class AWGNChannel(nn.Module):
    """
    Additive White Gaussian Noise channel.

    Adds noise calibrated to a target SNR (dB).
    Signal power is measured from the input — so the
    noise level adapts to whatever the encoder outputs.

    This is the standalone channel used for evaluation.
    The differentiable version for training is ChannelLayer.
    """

    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db     = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)

    def set_snr(self, snr_db: float):
        self.snr_db     = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z     : transmitted signal, shape (batch, bottleneck_dim)
        return: received signal with AWGN noise, same shape
        """
        signal_power = z.detach().pow(2).mean()
        noise_std    = torch.sqrt(signal_power / self.snr_linear)
        noise        = torch.randn_like(z) * noise_std
        return z + noise