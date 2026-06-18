import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============ Actual (Experimental) values ============

# ---- External Surface Roughness (ESR) ----
esr_actual = np.array([
    6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
    4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
    4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69
])

# ---- Material Removal Quantity (MRQ) ----
mrq_actual = np.array([
    100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
    66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
    86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57
])

# ---- Heat Affected Zone (HAZ) ----
haz_actual = np.array([
    90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
    74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
    75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59
])

# ============ Predicted (Ensemble) values ============

esr_pred = np.array([
    6.054, 4.093, 6.795, 5.116, 6.067, 5.807, 6.068, 5.566, 4.259, 5.295, 6.179, 7.227, 4.984, 5.707
    , 6.323, 5.916, 4.394, 5.886, 6.072, 7.057, 4.788, 6.629, 6.412, 4.701
])

mrq_pred = np.array([
    101.16, 77.37, 120.03, 80.90, 95.76, 99.76, 114.76, 95.06, 70.94, 85.12, 108.39, 116.43,
    91.69, 103.77, 112.34, 101.20, 85.50, 105.68, 106.52, 117.09, 87.33, 115.22, 107.58, 79.59,
])

haz_pred = np.array([
    90.337, 72.734, 101.499, 84.786, 90.456, 87.866, 90.365, 87.044, 75.572, 86.161, 92.538, 105.676, 83.136,
    86.988, 93.153, 88.969, 76.033, 88.999, 90.520, 103.685, 79.428, 96.158, 94.727, 81.066,
])

# ============ Performance metrics ============
targets = {
    "External Surface Roughness": (esr_actual, esr_pred),
    "Material Removal Quantity": (mrq_actual, mrq_pred),
    "Heat Affected Zone": (haz_actual, haz_pred)
}

performance_metrics = {}
for name, (y_true, y_pred) in targets.items():
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    performance_metrics[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

# ============ Plot function ============
def plot_actual_vs_pred(y_true, y_pred, title, metrics):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, edgecolor="k", alpha=0.7, s=60)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Ideal Fit")

    ax.set_title(f"{title} – Actual vs Predicted", fontsize=16, fontweight="bold")
    ax.set_xlabel("Actual Values", fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted Values", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold")
    ax.tick_params(axis='both', labelsize=12)

    metrics_text = (
        f"RMSE: {metrics['RMSE']:.4f}\n"
        f"MAE: {metrics['MAE']:.4f}\n"
        f"R²: {metrics['R²']:.4f}"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============ Plots ============
plot_actual_vs_pred(esr_actual, esr_pred, "External Surface Roughness", performance_metrics["External Surface Roughness"])
plot_actual_vs_pred(mrq_actual, mrq_pred, "Material Removal Quantity", performance_metrics["Material Removal Quantity"])
plot_actual_vs_pred(haz_actual, haz_pred, "HAZ", performance_metrics["Heat Affected Zone"])

# ============ Summary ============
print("\nEnsemble Model Performance Summary")
print("=" * 65)
print(f"{'Target':<30} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 65)
for name, m in performance_metrics.items():
    print(f"{name:<30} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {m['R²']:<10.4f}")
print("=" * 65)