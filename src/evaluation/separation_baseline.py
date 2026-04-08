import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.special import erfc
import sys
sys.path.append("/lake/workspaces/subash_ws/scs_robot")
from src.data.occupancy import get_dataloaders

#region Separation Baseline Classifier

class SeparationClassifier(nn.Module):
    """
    Simple CNN classifier that operates on
    reconstructed (noisy) occupancy grids.
    Represents the decoder side of a separation system.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)
        
        
def quantize_and_corrupt(grid, n_bits, snr_db):
    """
    Simulates separation pipeline:

    1. Quantise each pixel to n_bits levels
       (compression — reduces information to fixed bits)

    2. Add bit errors based on SNR
       (channel — bits get flipped with probability p_error)

    3. Reconstruct grid from corrupted bits
       (decompression — rebuild approximate grid)

    Args:
        grid    : (batch, 1, 64, 64) float tensor
        n_bits  : bits per pixel — controls compression ratio
        snr_db  : channel SNR in dB — controls error rate

    Returns:
        corrupted grid of same shape
    """
        
    batch_size = grid.size(0)

    # -- step1 - Quantize --

    levels = 2 ** n_bits
    grid_np = grid.cpu().numpy()
    quantized = np.round(grid_np * (levels - 1)) / (levels - 1)

    # -- step2 - Compute bit error probability from SNR --

    # For BPSK modulation: BER = Q(sqrt(2 * SNR_linear))
    # Q(x) ≈ 0.5 * erfc(x / sqrt(2))

    snr_linear = 10 ** (snr_db/10.0)
    ber = 0.5 * erfc(np.sqrt(snr_linear))
    ber = np.clip(ber, 0.0, 0.5)

    # -- step3 - Flip bits randomly based on BER --
    noise_mask = np.random.binomial(1,ber,quantized.shape)
    step = 1.0 / (levels - 1) if levels > 1 else 1.0
    corrupted = np.clip(quantized + noise_mask * step, 0.0, 1.0)

    return torch.tensor(corrupted, dtype=torch.float32)

def train_separation_baseline(n_bits=1, epochs=50, lr=1e-3):
    """
    Train the separation classifier on clean grids.
    At test time we corrupt the grids to simulate the channel.

    n_bits=1 means 1 bit per pixel = 4096 bits total for 64x64 grid.
    But we want fair comparison with our system which uses 64 bits.
    So we downsample the grid first to match bandwidth.

    For simplicity: n_bits controls quantisation quality.
    Lower n_bits = more compression = harder task.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SeparationClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dl, val_dl, _ = get_dataloaders(
        n_train=8000, n_val=1000, grid_size=64, batch_size=64
    )

    for epoch in range(epochs):
        model.train()
        for X, Y in train_dl:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = criterion(pred, Y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X, Y in val_dl:
                    X, Y = X.to(device), Y.to(device)
                    pred = model(X)
                    preds = pred.argmax(dim=1)
                    correct += (preds == Y.long()). sum(). item()
                    total += Y.size(0)
            print(f"  Epoch {epoch+1:3d} | Val Acc: {correct/total:.4f}")

    return model

def evaluate_separation(model, snr_values, n_bits=1):
    """
    Evaluate separation baseline at multiple SNR values.
    Corrupts grids before classification to simulate channel.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    _, val_dl, _ = get_dataloaders(
        n_val=1000, grid_size=64, batch_size=64
    )

    results = {}

    for snr_db in snr_values:
        correct, total = 0, 0

        with torch.no_grad():
            for X, Y in val_dl:
                Y = Y.to(device)

                # Corrupt grid to simulate separation channel
                X_corrupted = quantize_and_corrupt(X, n_bits, snr_db)
                X_corrupted = X_corrupted.to(device)

                pred = model(X_corrupted)
                preds = pred.argmax(dim=1)
                correct += (preds == Y.long()).sum().item()
                total += Y.size(0)

        acc = correct/total
        results[snr_db] = acc
        print(f"SNR = {snr_db:4d} dB  →  Acc: {acc:.4f}")

    return results

def random_baseline(n_classes=4):
    """Theoretical random baseline — always 1/n_classes."""
    return 1.0 / n_classes

# -- Main--

if __name__ == "__main__":
    
    snr_values = [-10, -5, 0, 5, 10, 15, 20]

    #train separation classifier on clean data
    model = train_separation_baseline(n_bits=1, epochs=50, lr=1e-3)

    #Evaluate at all SNR values
    sep_results = evaluate_separation(model, snr_values, n_bits=1)

    semantic_acc = [0.313, 0.330, 0.621, 0.859, 0.961, 0.987, 0.995]
    separation_acc = [sep_results[s] for s in snr_values]
    random_acc = [0.25] * len(snr_values)

    # Print comparison table
    print("\n" + "=" * 55)
    print(f"{'SNR':>6} | {'Semantic JSCC':>14} | {'Separation':>10} | {'Random':>6}")
    print("-" * 55)
    for i, snr in enumerate(snr_values):
        print(f"{snr:>6} | {semantic_acc[i]:>14.4f} | "
              f"{separation_acc[i]:>10.4f} | {random_acc[i]:>6.4f}")
    print("=" * 55)

    #save results
    os.makedirs('/lake/workspaces/subash_ws/scs_robot/results', exist_ok=True)
    results = {
        "snr_values": snr_values,
        "semantic_acc": semantic_acc,
        "separation_acc": separation_acc,
        "random_acc": random_acc
    }
    with open('/lake/workspaces/subash_ws/scs_robot/results/baseline_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/baseline_comparison.json")

    #plot comparison

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(snr_values, semantic_acc,   'b-o', linewidth=2,
            markersize=6, label='Semantic JSCC (proposed)')
    ax.plot(snr_values, separation_acc, 'r-s', linewidth=2,
            markersize=6, label='Separation baseline')
    ax.axhline(0.25, color='gray', linestyle=':',
               linewidth=1.5, label='Random (0.25)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Task Accuracy')
    ax.set_title('Semantic JSCC vs Separation Baseline')
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(snr_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/lake/workspaces/subash_ws/scs_robot/results/baseline_comparison.png',
                bbox_inches='tight', dpi=300)
    plt.savefig('/lake/workspaces/subash_ws/scs_robot/results/baseline_comparison.pdf',
                bbox_inches='tight')
    print("Saved to results/baseline_comparison.png")
    plt.show()

#endregion