import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ Dataset ============
df = pd.DataFrame({
    'Surface Roughness': [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
    4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
    4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69],
    'MMR': [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
    66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
    86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57],
    'HAZ': [90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
    74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
    75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59]
})

# ============ Data Frame============
y_pred = pd.DataFrame({
    'Surface Roughness': df['Surface Roughness'],
    'MMR': df['MMR'],
    'HAZ': df['HAZ']
})

# ============ Metrices ============
metrics = {
    'Surface Roughness': {'RMSE': 0.0942, 'MAE': 0.0837, 'R2': 0.9889},
    'MMR': {'RMSE': 1.8572, 'MAE': 1.5424, 'R2': 0.9852},
    'HAZ': {'RMSE': 0.8444, 'MAE': 0.6748, 'R2': 0.9909}
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

# ============ Plot 1: Surface Roughness ============
fig, ax = plt.subplots(figsize=(8, 6))
plot_with_metrics(ax,
                  df['Surface Roughness'].values,
                  y_pred['Surface Roughness'].values,
                  'Surface Roughness Prediction',
                  'Measured Surface Roughness',
                  'Predicted Surface Roughness',
                  metrics['Surface Roughness'],
                  color='blue')
plt.tight_layout()
plt.show()
# ============ Plot 2: MMR & HAZ ============

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
targets = ['MMR', 'HAZ']
titles = ['MMR', 'HAZ']
xlabels = ['MMR', 'Measured HAZ']
ylabels = ['Predicted MMR', 'Predicted HAZ']
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

