import matplotlib.pyplot as plt
import os

snr_values    = [-10, -5, 0, 5, 10, 15, 20]
awgn_acc      = [0.313, 0.330, 0.621, 0.859, 0.961, 0.987, 0.995]
rayleigh_acc  = [0.303, 0.303, 0.477, 0.699, 0.827, 0.921, 0.949]
random_acc    = [0.25] * len(snr_values)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(snr_values, awgn_acc,     'b-o', linewidth=2,
        markersize=6, label='Semantic JSCC — AWGN')
ax.plot(snr_values, rayleigh_acc, 'g-s', linewidth=2,
        markersize=6, label='Semantic JSCC — Rayleigh')
ax.axhline(0.25, color='gray', linestyle=':',
           linewidth=1.5, label='Random (0.25)')

ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Task Accuracy')
ax.set_title('Semantic JSCC — AWGN vs Rayleigh Fading')
ax.set_ylim(0.0, 1.05)
ax.set_xticks(snr_values)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('/lake/workspaces/subash_ws/scs_robot/results', exist_ok=True)
plt.savefig('/lake/workspaces/subash_ws/scs_robot/results/snr_sweep.png',
            bbox_inches='tight', dpi=300)
plt.savefig('/lake/workspaces/subash_ws/scs_robot/results/snr_sweep.pdf',
            bbox_inches='tight')
print("Saved to results/snr_sweep.png")
plt.show()