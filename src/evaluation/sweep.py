import matplotlib.pyplot as plt
import os

bottleneck_dims = [1, 2, 4, 8, 16]
bits_transmitted = [k * 32 for k in bottleneck_dims]

# Fill these in after runs complete
val_acc = [0.546, 0.82, 0.936, 0.976, 0.982]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Plot 1 — Accuracy vs bottleneck dimension
axes[0].plot(bottleneck_dims, val_acc, 'g-D',
             linewidth=2, markersize=7)
axes[0].set_xlabel('Bottleneck Dimension k (floats)')
axes[0].set_ylabel('Task Accuracy')
axes[0].set_title('Accuracy vs Bottleneck Size')
axes[0].set_xticks(bottleneck_dims)
axes[0].axhline(0.25, color='gray', linestyle=':',
                linewidth=1.5, label='Random (0.25)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2 — Accuracy vs bits transmitted
axes[1].semilogx(bits_transmitted, val_acc, 'g-D',
                 linewidth=2, markersize=7)
axes[1].set_xlabel('Bits Transmitted (log scale)')
axes[1].set_ylabel('Task Accuracy')
axes[1].set_title('Accuracy vs Bandwidth')
axes[1].axhline(0.25, color='gray', linestyle=':',
                linewidth=1.5, label='Random (0.25)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

for i, k in enumerate(bottleneck_dims):
    axes[1].annotate(f'k={k}',
                     (bits_transmitted[i], val_acc[i]),
                     textcoords="offset points",
                     xytext=(5, 5), fontsize=8)

plt.tight_layout()
os.makedirs('/lake/workspaces/subash_ws/scs_robot/results', exist_ok=True)
plt.savefig('/lake/workspaces/subash_ws/scs_robot/results/bottleneck_sweep.png',
            bbox_inches='tight', dpi=300)
plt.savefig('/lake/workspaces/subash_ws/scs_robot/results/bottleneck_sweep.pdf',
            bbox_inches='tight')
print("Saved to results/bottleneck_sweep.png")
plt.show()