import matplotlib.pyplot as plt
import os

snr_values = [-10, -5, 0, 5, 10, 15, 20]
val_acc    = [0.313, 0.330, 0.621, 0.859, 0.961, 0.987, 0.995]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(snr_values, val_acc, 'b-o', linewidth=2, markersize=6, label='Semantic JSCC')
ax.axhline(0.25, color='gray', linestyle=':', linewidth=1.5, label='Random baseline')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Task Accuracy')
ax.set_title('Navigation Accuracy vs Channel SNR')
ax.set_ylim(0.0, 1.05)
ax.set_xticks(snr_values)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('/lake/workspaces/subash_ws/scs_robot/results', exist_ok=True)
plt.savefig('/lake/workspaces/subash_ws/scs_robot/results/snr_sweep.png', bbox_inches='tight')
plt.savefig('/lake/workspaces/subash_ws/scs_robot/results/snr_sweep.pdf', bbox_inches='tight')
print("Saved to /lake/workspaces/subash_ws/scs_robot/results/")
plt.show()
