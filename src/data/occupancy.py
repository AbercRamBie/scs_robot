import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class OccupancyGridDataset(Dataset):
    """
    Procedurally generated 2D occupancy grids.

    Each sample is a 64x64 binary map:
        0 = free space
        1 = occupied (wall / obstacle)

    Label Y:
        1 = path ahead is clear (safe)
        0 = path ahead is blocked (obstacle present)

    The robot is always at the centre of the map.
    'Path ahead' = a corridor of width 5 cells directly
    above the centre, extending to the top edge.
    If that corridor is fully free → label 1, else label 0.
    """

    def __init__(self,
                 n_samples:  int = 10000,
                 grid_size:  int = 64,
                 min_rooms:  int = 3,
                 max_rooms:  int = 8,
                 seed:       int = 42):

        self.n_samples = n_samples
        self.grid_size = grid_size
        rng            = np.random.default_rng(seed)

        grids  = []
        labels = []

        for _ in range(n_samples):
            grid  = self._generate_map(rng, grid_size, min_rooms, max_rooms)
            label = self._compute_label(grid, grid_size)
            grids.append(grid)
            labels.append(label)

        # Shape: (N, 1, 64, 64) — channel-first for CNN
        self.grids  = torch.tensor(
            np.stack(grids), dtype=torch.float32
        ).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    # ── Map generation ──────────────────────────────────────────

    def _generate_map(self, rng, size, min_rooms, max_rooms):
        """
        Simple room-and-corridor procedural map.
        Start with all walls, carve out rooms and corridors.
        """
        grid = np.ones((size, size), dtype=np.float32)  # all walls

        # Always carve a free space around robot position (centre)
        cx, cy = size // 2, size // 2
        grid[cx-3:cx+4, cy-3:cy+4] = 0

        n_rooms = rng.integers(min_rooms, max_rooms + 1)

        room_centres = []
        for _ in range(n_rooms):
            # Random room position and size
            rw   = rng.integers(6, 16)
            rh   = rng.integers(6, 16)
            rx   = rng.integers(1, size - rw - 1)
            ry   = rng.integers(1, size - rh - 1)
            grid[rx:rx+rw, ry:ry+rh] = 0
            room_centres.append((rx + rw//2, ry + rh//2))

        # Connect rooms with corridors (L-shaped paths)
        for i in range(len(room_centres) - 1):
            x1, y1 = room_centres[i]
            x2, y2 = room_centres[i+1]
            # Horizontal then vertical
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            grid[min_x:max_x+1, y1] = 0
            grid[x2, min_y:max_y+1] = 0

        # Randomly add scattered obstacles in free space
        n_obstacles = rng.integers(5, 20)
        for _ in range(n_obstacles):
            ox = rng.integers(1, size - 2)
            oy = rng.integers(1, size - 2)
            if grid[ox, oy] == 0:            # only in free space
                grid[ox:ox+2, oy:oy+2] = 1  # small 2x2 block

        # Ensure border is always wall
        grid[0,  :]  = 1
        grid[-1, :]  = 1
        grid[:,  0]  = 1
        grid[:, -1]  = 1

        return grid

    def _compute_label(self, grid, size):
        """
        Check if the forward corridor (above robot centre) is clear.
        Corridor: width 5, from centre to top edge.
        Label 1 = clear, 0 = blocked.
        """
        cx   = size // 2
        cy   = size // 2
        half = 2   # half-width of corridor

        corridor = grid[:cy, cx-half:cx+half+1]  # above centre
        return float(corridor.sum() == 0)         # 1 if fully free

    # ── Dataset interface ───────────────────────────────────────

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.grids[idx], self.labels[idx]


def get_dataloaders(n_train:    int = 8000,
                    n_val:      int = 1000,
                    n_test:     int = 1000,
                    grid_size:  int = 64,
                    batch_size: int = 64):
    """
    Returns train, val, test dataloaders.
    Different seeds so splits don't overlap.
    """
    train_ds = OccupancyGridDataset(n_train, grid_size, seed=42)
    val_ds   = OccupancyGridDataset(n_val,   grid_size, seed=99)
    test_ds  = OccupancyGridDataset(n_test,  grid_size, seed=777)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size,
                          shuffle=False, num_workers=2)

    return train_dl, val_dl, test_dl