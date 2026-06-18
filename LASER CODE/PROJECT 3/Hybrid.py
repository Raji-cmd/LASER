import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ Dataset ============
df = pd.DataFrame({
    'Surface Undulation': [ 5.979, 4.191, 6.791, 5.173, 6.087, 5.733, 6.086, 5.498,
        4.384, 5.35, 6.1, 7.18, 5.054, 5.82, 6.292, 5.835,
        4.45, 5.809, 5.988, 6.991, 4.856, 6.644, 6.395, 4.731],
    'MRR': [ 100.401, 79.131, 120.923, 82.47, 98.176, 98.832, 113.374, 94.31,
        73.706, 87.317, 106.071, 117.261, 92.83, 105.097, 110.057, 100.027,
        85.243, 104.087, 104.521, 113.161, 87.554, 116.303, 108.177, 80.311],
    'HAZ': [ 90.228, 73.02, 100.662, 84.913, 91.161, 87.43, 90.805, 86.026,
        76.361, 87.244, 91.945, 104.26, 83.62, 87.424, 93.065, 88.651,
        76.144, 88.5, 90.263, 101.782, 80.061, 96.977, 95.142, 80.711]
})

# ============ Data Frame============
y_pred = pd.DataFrame({
    'Surface Undulation': df['Surface Undulation'],
    'MRR': df['MRR'],
    'HAZ': df['HAZ']
})

# ============ Metrices ============
metrics = {
    'Surface Undulation': {'RMSE': 0.1422, 'MAE': 0.1270, 'R2': 0.9747},
    'MRR': {'RMSE': 3.7230, 'MAE': 2.9948, 'R2': 0.9406},
    'HAZ': {'RMSE': 1.7036, 'MAE':1.3375, 'R2': 0.9629}
}

# ============ Font ============
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============ Function For Plotting ============
def plot_with_metrics(ax, y_true, y_hat, title, x_label, y_label, metric_dict, color='blue'):
    ax.scatter(y_true, y_hat, alpha=0.7, color=color, edgecolors='k', s=60)

    min_val, max_val = min(y_true) * 0.95, max(y_true) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')

    residuals = y_true - y_hat
    std = np.std(residuals)
    pi_upper = y_hat + 1.96 * std
    pi_lower = y_hat - 1.96 * std
    sort_idx = np.argsort(y_true)
    ax.fill_between(np.array(y_true)[sort_idx], pi_lower[sort_idx], pi_upper[sort_idx],
                    color='gray', alpha=0.3, label='95% Prediction Interval')

    # ============ Title and Axis Label ============
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    textstr = f"R² = {metric_dict['R2']:.4f}\nRMSE = {metric_dict['RMSE']:.4f}\nMAE = {metric_dict['MAE']:.4f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============ Plot 1: Surface Undulation ============
fig, ax = plt.subplots(figsize=(8, 6))
plot_with_metrics(ax,
                  df['Surface Undulation'].values,
                  y_pred['Surface Undulation'].values,
                  'Surface Undulation Prediction',
                  'Measured Surface Undulation',
                  'Predicted Surface Undulation',
                  metrics['Surface Undulation'],
                  color='blue')
plt.tight_layout()
plt.show()
# ============ Plot 2: MMR & HAZ ============

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
targets = ['MRR', 'HAZ']
titles = ['MRR', 'HAZ']
xlabels = ['MRR', 'Measured HAZ']
ylabels = ['Predicted MRR', 'Predicted HAZ']
colors = ['green', 'purple']

for i, ax in enumerate(axes):
    plot_with_metrics(ax,
                      df[targets[i]].values,
                      y_pred[targets[i]].values,
                      titles[i],
                      xlabels[i],
                      ylabels[i],
                      metrics[targets[i]],
                      color=colors[i])

plt.tight_layout()
plt.show()

