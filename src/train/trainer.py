import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from src.models.encoder       import SemanticEncoder
from src.models.decoder       import SemanticDecoder
from src.models.channel_layer import ChannelLayer
from src.loss.vib             import vib_loss, reparametrize
from src.data.occupancy       import get_dataloaders
from src.train.config         import Config

def train(cfg: Config):

    # ── Setup ─────────────────────────────────────────────────
    wandb.init(project=cfg.project_name,
               name=cfg.run_name,
               config=cfg.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"SNR    : {cfg.snr_db_train} dB")
    print(f"Beta   : {cfg.beta}")
    print(f"Bottleneck dim: {cfg.bottleneck_dim}")

    # ── Models ────────────────────────────────────────────────
    encoder = SemanticEncoder(bottleneck_dim=cfg.bottleneck_dim).to(device)
    decoder = SemanticDecoder(bottleneck_dim=cfg.bottleneck_dim,
                              hidden_dims=cfg.decoder_hidden).to(device)
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

    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

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

                preds    = (Y_pred.squeeze(1) > 0).float()
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

        print(f"Epoch {epoch+1:03d} | "
              f"Loss: {metrics['loss']/n_bat:.4f} | "
              f"Task: {metrics['task_loss']/n_bat:.4f} | "
              f"KL: {metrics['kl_loss']/n_bat:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    wandb.finish()

    return encoder, decoder, channel