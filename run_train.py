from src.train.trainer import train
from src.train.config  import Config

cfg = Config(
    snr_db_train = 10.0,
    beta         = 1e-3,
    bottleneck_dim = 16,
    epochs       = 30,
    run_name     = "occupancy_awgn_snr10_beta1e3"
)

encoder, decoder, channel = train(cfg)