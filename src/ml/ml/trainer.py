import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from encoder import SemanticEncoder
from decoder import SemanticDecoder
from channel_layer import ChannelLayer
from vib import vib_loss, reparametrize
from occupancy import get_dataloaders
from config import Config
from rayleigh import RayleighChannel

def train(cfg: Config):

    # ── Setup ─────────────────────────────────────────────────
    wandb.init(project=cfg.project_name,
               name=cfg.run_name,
               config=cfg.__dict__,
               dir="./artifacts/wandb")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Models ────────────────────────────────────────────────
    encoder = SemanticEncoder(bottleneck_dim=cfg.bottleneck_dim).to(device)
    decoder = SemanticDecoder(bottleneck_dim=cfg.bottleneck_dim,
                              hidden_dims=cfg.decoder_hidden).to(device)
    if cfg.channel_type == "rayleigh":
       channel = RayleighChannel(snr_db=cfg.snr_db_train).to(device)
    else:
       channel = ChannelLayer(snr_db=cfg.snr_db_train).to(device)
    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr)

    # ── Data ──────────────────────────────────────────────────
    train_dl, val_dl, _ = get_dataloaders(
        n_train    = cfg.n_train,
        n_val      = cfg.n_val,
        grid_size  = cfg.grid_size,
        batch_size = cfg.batch_size
    )
    best_val_acc = 0.0

    for epoch in range(cfg.epochs):

        # ── Train ─────────────────────────────────────────────
        encoder.train()
        decoder.train()
        metrics = {"loss": 0.0, "task_loss": 0.0, "kl_loss": 0.0}

        for X, Y in tqdm(train_dl, desc=f"Epoch {epoch+1:03d}",
                         leave=False):
            X, Y = X.to(device), Y.to(device)

            # Forward pass
            mu, log_var = encoder(X)
            Z           = reparametrize(mu, log_var)
            Z_hat       = channel(Z)
            Y_pred      = decoder(Z_hat)

            # Loss
            losses = vib_loss(Y_pred, Y, mu, log_var, cfg.beta)

            # Backward
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            for k in metrics:
                metrics[k] += losses[k].item()

        # ── Validate ──────────────────────────────────────────
        encoder.eval()
        decoder.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for X, Y in val_dl:
                X, Y        = X.to(device), Y.to(device)
                mu, log_var = encoder(X)
                Z           = reparametrize(mu, log_var)
                Z_hat       = channel(Z)
                Y_pred      = decoder(Z_hat)

                preds    = Y_pred.argmax(dim=1)
                correct += (preds == Y).sum().item()
                total   += Y.size(0)
        val_acc = correct / total
        n_bat   = len(train_dl)

        # ── Log ───────────────────────────────────────────────
        log = {
            "epoch":           epoch + 1,
            "val_acc":         val_acc,
            "train/loss":      metrics["loss"]      / n_bat,
            "train/task_loss": metrics["task_loss"] / n_bat,
            "train/kl_loss":   metrics["kl_loss"]   / n_bat,
            "snr_db":          cfg.snr_db_train,
            "beta":            cfg.beta,
        }
        wandb.log(log)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    wandb.finish()

    return encoder, decoder, channel