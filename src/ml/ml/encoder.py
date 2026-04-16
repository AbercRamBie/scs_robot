import torch
import torch.nn as nn


class SemanticEncoder(nn.Module):
    """
    CNN semantic encoder.

    Input  : occupancy grid (batch, 1, 64, 64)
    Output : mu and log_var of bottleneck distribution q(Z|X)
             both shape (batch, bottleneck_dim)

    The encoder does NOT sample Z — that happens in the
    training loop via the reparametrisation trick.
    This keeps the sampling logic visible and explicit.

    Architecture:
        Conv layers extract spatial features from the grid.
        Flatten + Linear layers map to the bottleneck.
    """

    def __init__(self,
                 bottleneck_dim: int = 16,
                 in_channels:    int = 1):
        super().__init__()

        # ── Convolutional feature extractor ─────────────────
        # Input: (batch, 1, 64, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),                    # → (batch, 32, 32, 32)

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),                    # → (batch, 64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),                    # → (batch, 128, 8, 8)

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),                    # → (batch, 256, 4, 4)
        )

        # Flattened size after conv: 256 * 4 * 4 = 4096
        self.flat_dim = 256 * 4 * 4

        # ── Bottleneck projection ────────────────────────────
        self.fc_shared  = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU()
        )
        self.fc_mu      = nn.Linear(512, bottleneck_dim)
        self.fc_log_var = nn.Linear(512, bottleneck_dim)

    def forward(self, x: torch.Tensor):
        """
        x : (batch, 1, 64, 64)
        returns: mu (batch, k), log_var (batch, k)
        """
        h       = self.conv(x)
        h       = h.view(h.size(0), -1)      # flatten
        h       = self.fc_shared(h)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var