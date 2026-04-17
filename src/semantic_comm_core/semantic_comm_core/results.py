import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

results_dir = '/lake/workspaces/subash_ws/scs_robot/results'
os.makedirs(results_dir, exist_ok=True)

#region Data

snr_values   = [-10, -5, 0, 5, 10, 15, 20]
snr_acc      = [0.313, 0.330, 0.621, 0.859, 0.961, 0.987, 0.995]

beta_values  = [0.001, 0.01, 0.1, 0.5, 1.0]
kl_values    = [362.8, 36.3, 4.2, 1.1, 0.5]
beta_acc     = [0.948, 0.950, 0.925, 0.862, 0.735]

mismatch_snr = [-10, -5, 0, 5, 10, 15, 20]
mismatch_acc = [0.383, 0.484, 0.654, 0.820, 0.936, 0.972, 0.982]

#endregion

#region figure

fig = plt.figure(figsize=(14, 4))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

#endregion

# region Plot 1 - snr_sweep

ax1 = fig.add_subplot(gs[0])
ax1.plot(snr_values, snr_acc, 'b-o', linewidth=2,
         markersize=6, label='Semantic JSCC')
ax1.axhline(0.25, color='gray', linestyle=':',
            linewidth=1.5, label='Random (0.25)')
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('Task Accuracy')
ax1.set_title('(a) SNR Sweep')
ax1.set_ylim(0.0, 1.05)
ax1.set_xticks(snr_values)
ax1.tick_params(axis='x', labelsize=7)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('/lake/workspaces/subash_ws/scs_robot/results', exist_ok=True)
plt.savefig('/lake/workspaces/subash_ws/scs_robot/results/snr_sweep.png', bbox_inches='tight')
plt.savefig('/lake/workspaces/subash_ws/scs_robot/results/snr_sweep.pdf', bbox_inches='tight')
plt.show()

# endregion

#region plot - IB curve

ax2 = fig.add_subplot(gs[1])
ax2.plot(kl_values, beta_acc, 'r-s', linewidth=2,
         markersize=6)
ax2.set_xlabel('KL Divergence  I(Z;X)')
ax2.set_ylabel('Task Accuracy  I(Z;Y)')
ax2.set_title('(b) Information Bottleneck Curve')
ax2.grid(True, alpha=0.3)
ax2.axhline(0.25, color='gray', linestyle=':', linewidth=1.5)

for i, b in enumerate(beta_values):
    ax2.annotate(f'β={b}',
                 (kl_values[i], beta_acc[i]),
                 textcoords="offset points",
                 xytext=(5, 5), fontsize=7)

#endregion

#region plot - Channel Mismatch

ax3 = fig.add_subplot(gs[2])
ax3.plot(mismatch_snr, mismatch_acc, 'g-^', linewidth=2,
         markersize=6, label='Semantic JSCC (train SNR=10)')
ax3.plot(snr_values, snr_acc, 'b--o', linewidth=1.5,
         markersize=5, alpha=0.6, label='Matched training')
ax3.axvline(10, color='orange', linestyle='--',
            linewidth=1.5, label='Train SNR')
ax3.axhline(0.25, color='gray', linestyle=':',
            linewidth=1.5, label='Random (0.25)')
ax3.set_xlabel('Test SNR (dB)')
ax3.set_ylabel('Task Accuracy')
ax3.set_title('(c) Channel Mismatch')
ax3.set_ylim(0.0, 1.05)
ax3.set_xticks(mismatch_snr)
ax3.tick_params(axis='x', labelsize=7)
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)

#endregion

plt.suptitle('Semantic JSCC for Robot Navigation — Key Results',
             fontsize=12, y=1.02)
plt.savefig(f'{results_dir}/all_results.png',
            bbox_inches='tight', dpi=300)
plt.savefig(f'{results_dir}/all_results.pdf',
            bbox_inches='tight')
plt.show()
