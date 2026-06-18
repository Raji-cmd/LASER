import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ Dataset ============
df = pd.DataFrame({
     "External Surface Roughness": [6.054, 4.093, 6.795, 5.116, 6.067, 5.807, 6.068, 5.566, 4.259, 5.295, 6.179, 7.227, 4.984, 5.707
        , 6.323, 5.916, 4.394, 5.886, 6.072, 7.057, 4.788, 6.629, 6.412, 4.701],

    "Material Removal Quantity": [101.16, 77.37, 120.03, 80.90, 95.76, 99.76, 114.76, 95.06, 70.94, 85.12, 108.39, 116.43,
         91.69, 103.77, 112.34, 101.20, 85.50, 105.68, 106.52, 117.09, 87.33, 115.22, 107.58, 79.59],

    "Heat Affected Zone": [90.337, 72.734, 101.499, 84.786, 90.456, 87.866, 90.365, 87.044, 75.572, 86.161, 92.538, 105.676, 83.136,
        86.988, 93.153, 88.969, 76.033, 88.999, 90.520, 103.685, 79.428, 96.158, 94.727, 81.066]
})

# ============ Data Frame============
y_pred = pd.DataFrame({
    'External Surface Roughness': df['External Surface Roughness'],
    'MRQ': df['MRQ'],
    'HAZ': df['HAZ']
})

# ============ Metrices ============
metrics = {
    'External Surface Roughness': {'RMSE': 0.0784, 'MAE': 0.0679, 'R2': 0.9923},
    'MRQ': {'RMSE': 2.0739, 'MAE': 1.5967, 'R2': 0.9816},
    'HAZ': {'RMSE': 0.9950, 'MAE':0.7673, 'R2': 0.9873}
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

# ============ Plot 1: External Surface Roughness ============
fig, ax = plt.subplots(figsize=(8, 6))
plot_with_metrics(ax,
                  df['External Surface Roughness'].values,
                  y_pred['External Surface Roughness'].values,
                  'External Surface Roughness',
                  'Measured External Surface Roughness',
                  'Predicted External Surface Roughness',
                  metrics['External Surface Roughness'],
                  color='blue')
plt.tight_layout()
plt.show()
# ============ Plot 2: MRQ & HAZ ============

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
targets = ['MRQ', 'HAZ']
titles = ['MRQ', 'HAZ']
xlabels = ['MRQ', 'Measured HAZ']
ylabels = ['Predicted MRQ', 'Predicted HAZ']
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

