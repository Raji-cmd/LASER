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
    "External Surface Roughness": [6.054, 4.093, 6.795, 5.116, 6.067, 5.807, 6.068, 5.566, 4.259, 5.295, 6.179, 7.227,
                                   4.984, 5.707
        , 6.323, 5.916, 4.394, 5.886, 6.072, 7.057, 4.788, 6.629, 6.412, 4.701],

    "Material Removal Quantity": [101.16, 77.37, 120.03, 80.90, 95.76, 99.76, 114.76, 95.06, 70.94, 85.12, 108.39,
                                  116.43,
                                  91.69, 103.77, 112.34, 101.20, 85.50, 105.68, 106.52, 117.09, 87.33, 115.22, 107.58,
                                  79.59],

    "Heat Affected Zone": [90.337, 72.734, 101.499, 84.786, 90.456, 87.866, 90.365, 87.044, 75.572, 86.161, 92.538,
                           105.676, 83.136,
                           86.988, 93.153, 88.969, 76.033, 88.999, 90.520, 103.685, 79.428, 96.158, 94.727, 81.066]
}

df = pd.DataFrame(data)

# ============ Font configuration ============
font_name = 'DejaVu Sans'
font_size = 14
title_size = 16
font_weight = 'bold'

# ============ Features and targets ============
X = df[['Laser Power','Scanning Speed','Number of passes','Laser Frequency']]
targets = ['External Surface Roughness', 'Material Removal Quantity', 'Heat Affected Zone']

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
