import torch
import torch.nn.functional as F

def reparametrize(mu: torch.Tensor,
                  log_var: torch.Tensor) -> torch.Tensor:
    """
    Reparametrisation trick:
        Z = mu + eps * std,   eps ~ N(0, I)

    Why this works: the randomness is in eps (not in Z directly)
    so gradients flow through mu and log_var to the encoder.
    """
    std = torch.exp(0.5 * log_var).clamp(max=10)  # clamp for stability
    eps = torch.randn_like(std)
    return mu + eps * std

def vib_loss(y_pred:  torch.Tensor,
             y_true:  torch.Tensor,
             mu:      torch.Tensor,
             log_var: torch.Tensor,
             beta:    float = 1e-3) -> dict:
    """
    Variational Information Bottleneck loss:

        L = -I(Z;Y)  +  beta * I(Z;X)
          ≈ TaskLoss  +  beta * KL[q(Z|X) || p(Z)]

    KL divergence (closed form for Gaussians, p(Z) = N(0,I)):
        KL = -0.5 * mean(1 + log_var - mu² - exp(log_var))

    Args:
        y_pred  : decoder output logits, shape (batch, 1)
        y_true  : binary labels,         shape (batch,)
        mu      : encoder mean,          shape (batch, k)
        log_var : encoder log variance,  shape (batch, k)
        beta    : IB tradeoff parameter

    Returns dict so every term is logged to wandb separately.

    y_pred : (batch, 4) logits
    y_true : (batch,)   long int class labels

    """
    # ── Task loss ────────────────────────────────────────────
    task_loss = F.cross_entropy(
        y_pred, y_true.long()
    )

    # ── KL divergence ────────────────────────────────────────
    kl_loss = -0.5 * torch.mean(
        1 + log_var - mu.pow(2) - log_var.exp()
    )

    # ── Total VIB loss ───────────────────────────────────────
    total = task_loss + beta * kl_loss

    return {
        "loss":      total,
        "task_loss": task_loss.detach(),
        "kl_loss":   kl_loss.detach(),
    }