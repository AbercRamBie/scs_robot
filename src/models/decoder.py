import torch
import torch.nn as nn


class SemanticDecoder(nn.Module):
    """
    MLP task decoder.

    Input  : received bottleneck Ẑ (batch, bottleneck_dim)
    Output : task logit (batch, 1)
             raw score before sigmoid — use BCEWithLogitsLoss

    For binary classification (path clear / blocked).
    To extend to regression: change output_dim and loss.
    """

    def __init__(self,
                 bottleneck_dim: int = 16,
                 hidden_dims:    list = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev   = bottleneck_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))   # single logit output

        self.net = nn.Sequential(*layers)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        z_hat : (batch, bottleneck_dim)
        return: logit (batch, 1)
        """
        return self.net(z_hat)