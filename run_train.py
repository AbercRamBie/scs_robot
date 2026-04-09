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

bottleneck_values = [1, 2, 4, 8, 16]

for k in bottleneck_values:
    cfg = Config(
        snr_db_train = 5.0,
        beta = 0.5,
        bottleneck_dim = k,
        epochs = 100,
        lr = 3e-4,
        run_name = f"bottleneck_sweep_k{k}"
    )
    print(f"\nTraining with bottleneck_dim = {k}")
    train(cfg)
    

