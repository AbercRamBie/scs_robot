import torch
import os
import json
from src.train.trainer import train
from src.train.config  import Config
from src.data.occupancy import OccupancyGridDataset
from src.data.occupancy import get_dataloaders
from src.models.encoder import SemanticEncoder
from src.models.decoder import SemanticDecoder
from src.models.channel_layer import ChannelLayer
from src.loss.vib import reparametrize

cfg = Config(
        snr_db_train   = 10.0,
        beta           = 0.5,
        bottleneck_dim = 2,
        epochs         = 100,
        lr             = 3e-4,
        run_name       = "mismatch_train_snr10"
)

encoder, decoder, channel = train(cfg)
os.makedirs('checkpoints', exist_ok=True)
torch.save(encoder.state_dict(), 'checkpoints/encoder_snr10.pth')
torch.save(decoder.state_dict(), 'checkpoints/encoder_snr10.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.eval()
decoder.eval()

_, val_dl, _ = get_dataloaders(
    n_val      = 1000,
    grid_size  = cfg.grid_size,
    batch_size = cfg.batch_size
)

test_snrs = [-10, -5, 0, 5, 10, 15, 20]
mismatch_acc = []

for test_snr in test_snrs:
    channel.set_snr(test_snr)
    correct, total = 0,0
    with torch.no_grad():
        for X, Y in val_dl:
            X, Y = X.to(device), Y.to(device)
            mu, log_var = encoder(X)
            Z = reparametrize(mu, log_var)
            Z_hat = channel(Z)
            Y_pred = decoder(Z_hat)
            preds = Y_pred.argmax(dim=1)
            correct += (preds == Y.long()).sum().item()
            total += Y.size(0)

    acc = correct/total
    mismatch_acc.append(acc)
os.makedirs('results', exist_ok = True)

results = {
    "train_snr":    10,
    "test_snrs":    test_snrs,
    "mismatch_acc": mismatch_acc
}

with open('results/mismatch_results.json', 'w') as f:
    json.dump(results, f, indent=2)

