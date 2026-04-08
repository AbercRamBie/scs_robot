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
        self.labels = torch.tensor(labels, dtype=torch.long)

    # Map generation

    def _generate_map(self, rng, size, min_rooms, max_rooms):
        """
        Build map guaranteed to have at least one clear corridor.
        Robot at centre. Corridors carved explicitly in each direction.
        Obstacles placed only away from the main corridors.
        """
        grid = np.ones((size, size), dtype=np.float32)  # all walls

        # carve a free space around the robot position
        cx, cy = size // 2, size // 2
        grid[cx-4:cx+5, cy-4:cy+5] = 0

        open_forward = rng.random() > 0.7
        open_left = rng.random() > 0.65
        open_right = rng.random() > 0.45

        corridor_w = 3

        if open_forward:
            # Carve upward corridor from robot to top
            grid[0:cy, cx-corridor_w:cx+corridor_w+1] = 0

        if open_left:
            # Carve left corridor from robot to left edge
            grid[cy-corridor_w:cy+corridor_w+1, 0:cx] = 0 

        if open_right:
            # Carve right corridor from robot to right edge
            grid[cy-corridor_w:cy+corridor_w+1, cx:size] = 0
            # Add random rooms away from corridor centres
            
        n_rooms = rng.integers(2, 6)
        for _ in range(n_rooms):
           rw = rng.integers(5, 12)
           rh = rng.integers(5, 12)
           rx = rng.integers(1, size - rw - 1)
           ry = rng.integers(1, size - rh - 1)
           grid[rx:rx+rw, ry:ry+rh] = 0

        n_obstacles = rng.integers(10, 25)
        attempts = 0
        placed = 0

        while placed < n_obstacles and attempts < 200:
            ox = rng.integers(1, size - 3)
            oy = rng.integers(1, size - 3)
            attempts += 1

            in_forward = (cx-corridor_w-1 <= oy <= cx+corridor_w+1
                      and ox < cy - 4)
            in_left = (cy-corridor_w-1 <= ox <= cy+corridor_w+1
                      and oy < cx - 4)
            in_right = (cy-corridor_w-1 <= ox <= cy+corridor_w+1
                      and oy > cx + 4)
            in_robot = (cy-5 <= ox <= cy+5 and cx-5 <= oy <= cx+5)

            if not any([in_forward and open_forward,
                    in_left    and open_left,
                    in_right   and open_right,
                    in_robot]):
                grid[ox:ox+2, oy:oy+2] = 1
                placed += 1
                
        # Ensure border is always wall
        grid[0,  :]  = 1
        grid[-1, :]  = 1
        grid[:,  0]  = 1
        grid[:, -1]  = 1

        self._last_corridors = (open_forward, open_left, open_right)

        return grid

    def _compute_label(self, grid, size):
        """
        Use the corridor flags set during map generation.
        Forward takes priority, then left, then right, then stop.
        """
        open_forward, open_left, open_right = self._last_corridors

        if open_forward:
            return 0.0
        elif open_left:
            return 1.0
        elif open_right:
            return 2.0
        else:
            return 3.0

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