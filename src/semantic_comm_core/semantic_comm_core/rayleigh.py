import torch
import torch.nn as nn

class RayleighChannel(nn.Module):
    """
    Rayleigh fading channel.

    Models a wireless channel where signal amplitude varies
    randomly due to multipath propagation — more realistic
    than AWGN for a moving robot in an indoor environment.

    Two effects combined:
        1. Rayleigh fading  — random amplitude scaling
        2. AWGN             — additive Gaussian noise

    The fading coefficient h ~ Rayleigh(1/sqrt(2))
    is drawn independently for each transmission.
    """

    def __init__(self, snr_db: float=10.0):
        super().__init__()
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)

    def set_snr(self, snr_db: float):
        self.snr_db = snr_db
        self.snr_linear = 10**(snr_db / 10.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z     : transmitted signal (batch, bottleneck_dim)
        return: received signal with fading + noise
        """

        # Rayleigh fading coefficient
        # h = |h_real + j*h_imag| where h_real, h_imag ~ N(0, 1/2)
        h_real = torch.randn_like(z) * (1.0 / (2 ** 0.5))
        h_imag = torch.randn_like(z) * (1.0 / (2 ** 0.5))
        h_mag = torch.sqrt(h_real ** 2 + h_imag ** 2)

        z_faded = h_mag * z

        signal_power = z_faded.pow(2).mean().detach()
        noise_std = torch.sqrt(signal_power / self.snr_linear)
        noise = torch.randn_like(z_faded) * noise_std

        return z_faded + noise
