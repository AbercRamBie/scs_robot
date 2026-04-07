from dataclasses import dataclass, field

@dataclass
class Config:

    # ── Data ─────────────────────────────────────────────────
    grid_size:      int   = 64
    n_train:        int   = 8000
    n_val:          int   = 1000
    n_test:         int   = 1000
    batch_size:     int   = 64

    # ── Model ────────────────────────────────────────────────
    bottleneck_dim: int   = 16
    decoder_hidden: list  = field(default_factory=lambda: [128, 64])

    # ── Channel ──────────────────────────────────────────────
    snr_db_train:   float = 10.0

    # ── Training ─────────────────────────────────────────────
    beta:           float = 1e-3
    lr:             float = 1e-3
    epochs:         int   = 30

    # ── Logging ──────────────────────────────────────────────
    project_name:   str   = "semcomm-robot"
    run_name:       str   = "occupancy_awgn_snr10"