import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============ Actual (Experimental) values ============

# ---- Surface Undulation (SU) ----
su_actual = np.array([
    6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
    4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
    4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69
])

# ---- Material Removal Rate (MRR) ----
mrr_actual = np.array([
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

su_pred = np.array([
    5.979, 4.191, 6.791, 5.173, 6.087, 5.733, 6.086, 5.498,
    4.384, 5.35, 6.1, 7.18, 5.054, 5.82, 6.292, 5.835,
    4.45, 5.809, 5.988, 6.991, 4.856, 6.644, 6.395, 4.731
])

mrr_pred = np.array([
    100.401, 79.131, 120.923, 82.47, 98.176, 98.832, 113.374, 94.31,
    73.706, 87.317, 106.071, 117.261, 92.83, 105.097, 110.057, 100.027,
    85.243, 104.087, 104.521, 113.161, 87.554, 116.303, 108.177, 80.311
])

haz_pred = np.array([
    90.228, 73.02, 100.662, 84.913, 91.161, 87.43, 90.805, 86.026,
    76.361, 87.244, 91.945, 104.26, 83.62, 87.424, 93.065, 88.651,
    76.144, 88.5, 90.263, 101.782, 80.061, 96.977, 95.142, 80.711
])

# ============ Performance metrics ============
targets = {
    "Surface Undulation": (su_actual, su_pred),
    "Material Removal Rate": (mrr_actual, mrr_pred),
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
plot_actual_vs_pred(su_actual, su_pred, "Surface Undulation", performance_metrics["Surface Undulation"])
plot_actual_vs_pred(mrr_actual, mrr_pred, "Material Removal Rate", performance_metrics["Material Removal Rate"])
plot_actual_vs_pred(haz_actual, haz_pred, "HAZ", performance_metrics["Heat Affected Zone"])

# ============ Summary ============
print("\nEnsemble Model Performance Summary")
print("=" * 65)
print(f"{'Target':<30} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 65)
for name, m in performance_metrics.items():
    print(f"{name:<30} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {m['R²']:<10.4f}")
print("=" * 65)