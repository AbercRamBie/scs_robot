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

snr_values = [-10, -5, 0, 5, 10, 15, 20]

for snr in snr_values:
    cfg = Config(
        snr_db_train   = snr,
        beta           = 0.5,
        bottleneck_dim = 2,
        epochs         = 80,
        lr             = 3e-4,
        channel_type   = "rayleigh",
        run_name       = f"rayleigh_snr_sweep_{snr}dB"
    )
    print(f"\nTraining with Rayleigh channel at SNR = {snr} dB")
    train(cfg)


