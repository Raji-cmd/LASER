import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample

# ============ Dataset ============
data = {
    "Laser Power": [8.5, 6.5, 10.5, 6.5, 6.5, 8.5, 10.5, 8.5, 6.5, 10.5, 8.5, 10.5,
                    10.5, 6.5, 8.5, 7.5, 10.5, 8.5, 9.5, 6.5, 6.5, 6.5, 10.5, 10.5],
    "Scanning Speed": [280, 360, 200, 360, 200, 280, 200, 320, 360, 360, 280, 200,
                       360, 200, 240, 280, 360, 280, 280, 200, 360, 200, 200, 360],
    "Number of passes": [24, 12, 36, 36, 12, 18, 12, 24, 12, 36, 30, 36,
                         36, 12, 24, 24, 12, 24, 24, 36, 36, 36, 12, 12],
    "Laser Frequency": [48, 18, 18, 58, 58, 38, 18, 38, 58, 58, 38, 58,
                        18, 18, 38, 38, 18, 28, 38, 58, 18, 18, 58, 58],
    "Surface Undulation": [ 5.979, 4.191, 6.791, 5.173, 6.087, 5.733, 6.086, 5.498,
        4.384, 5.35, 6.1, 7.18, 5.054, 5.82, 6.292, 5.835,
        4.45, 5.809, 5.988, 6.991, 4.856, 6.644, 6.395, 4.731],
    "Material Removal Rate": [ 100.401, 79.131, 120.923, 82.47, 98.176, 98.832, 113.374, 94.31,
        73.706, 87.317, 106.071, 117.261, 92.83, 105.097, 110.057, 100.027,
        85.243, 104.087, 104.521, 113.161, 87.554, 116.303, 108.177, 80.311],
    "Heat Affected Zone": [ 90.228, 73.02, 100.662, 84.913, 91.161, 87.43, 90.805, 86.026,
        76.361, 87.244, 91.945, 104.26, 83.62, 87.424, 93.065, 88.651,
        76.144, 88.5, 90.263, 101.782, 80.061, 96.977, 95.142, 80.711]
}

df = pd.DataFrame(data)

# ============ Font configuration ============
font_name = 'DejaVu Sans'
font_size = 14
title_size = 16
font_weight = 'bold'

# ============ Features and targets ============
X = df[['Laser Power','Scanning Speed','Number of passes','Laser Frequency']]
targets = ['Surface Undulation', 'Material Removal Rate', 'Heat Affected Zone']

n_boot = 200

for target in targets:
    y = df[target]
    preds_boot = np.zeros((n_boot, len(X)))

    for i in range(n_boot):
        X_s, y_s = resample(X, y, replace=True, random_state=42 + i)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_s, y_s)
        preds_boot[i] = model.predict(X)

    lower = np.percentile(preds_boot, 2.5, axis=0)
    upper = np.percentile(preds_boot, 97.5, axis=0)
    median = np.median(preds_boot, axis=0)

    # ============ Title and Axis Label ============
    plt.figure(figsize=(12, 6))
    plt.fill_between(range(len(y)), lower, upper, alpha=0.3, color='gray', label="95% Prediction Interval")
    plt.plot(range(len(y)), y.values, "o-", color='blue', label="Actual", markersize=5, lw=2)
    plt.plot(range(len(y)), median, "s--", color='red', label="Median prediction", markersize=4, lw=2)
    plt.xlabel("Sample Index", fontname=font_name, fontsize=font_size, fontweight=font_weight)
    plt.ylabel(target.replace("_", " ").title(), fontname=font_name, fontsize=font_size, fontweight=font_weight)
    plt.title(f"Prediction Intervals (95%) - {target.replace('_', ' ').title()}",
              fontname=font_name, fontsize=16, fontweight=font_weight, pad=15)
    plt.legend(fontsize=font_size, prop={'family': font_name, 'weight': font_weight})
    plt.xticks(fontname=font_name, fontsize=12, fontweight=font_weight)
    plt.yticks(fontname=font_name, fontsize=12, fontweight=font_weight)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
