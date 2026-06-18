import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============ Actual (Experimental) values ============

# ---- Surface Waviness (SW) ----
sw_actual = np.array([
    6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
    4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
    4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69
])

# ---- Material Vaporization Rate (MVR) ----
mvr_actual = np.array([
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

sw_pred = np.array([
    6.058, 4.045, 6.831, 5.075, 6.090, 5.837, 6.086, 5.595,
    4.216, 5.269, 6.179, 7.253, 4.954, 5.688, 6.289, 5.945,
    4.361, 5.908, 6.083, 7.097, 4.757, 6.648, 6.438, 4.668
])

mvr_pred = np.array([
    101.242, 75.847, 120.486, 79.510, 95.451, 100.088, 115.970, 95.406,
    68.498, 83.970, 108.874, 116.749, 91.344, 103.544, 112.447, 101.373,
    85.379, 105.910, 106.919, 119.898, 87.092, 115.320, 107.522, 78.744
])

haz_pred = np.array([
    90.275, 72.287, 102.351, 84.596, 90.363, 88.070, 90.342, 87.704,
    75.089, 85.898, 92.626, 107.341, 82.966, 86.991, 92.841, 88.993,
    75.926, 89.110, 90.525, 105.305, 79.328, 96.297, 94.604, 81.024
])

# ============ Performance metrics ============

targets = {
    "Surface Waviness": (sw_actual, sw_pred),
    "Material Vaporization Rate": (mvr_actual, mvr_pred),
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
plot_actual_vs_pred(sw_actual, sw_pred, "Surface Waviness", performance_metrics["Surface Waviness"])
plot_actual_vs_pred(mvr_actual, mvr_pred, "MVR", performance_metrics["Material Vaporization Rate"])
plot_actual_vs_pred(haz_actual, haz_pred, "HAZ", performance_metrics["Heat Affected Zone"])

# ============ Summary ============
print("\nEnsemble Model Performance Summary")
print("=" * 65)
print(f"{'Target':<30} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 65)
for name, m in performance_metrics.items():
    print(f"{name:<30} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {m['R²']:<10.4f}")
print("=" * 65)
