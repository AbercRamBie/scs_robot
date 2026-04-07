from src.train.trainer import train
from src.train.config  import Config
from src.data.occupancy import OccupancyGridDataset

snr_values = [-10,-5,0,5,10,15,20]

for snr in snr_values:
    cfg = Config(
        snr_db_train   = snr,
        beta           = 0.5,
        bottleneck_dim = 2,
        epochs         = 100,
        lr             = 3e-4,
        run_name       = f"snr_sweep_{snr}dB"
    )
    print(f"\nTraining at SNR = {snr} dB")
    train(cfg)