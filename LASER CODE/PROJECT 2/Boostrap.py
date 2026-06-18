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
    "Surface Roughness": [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66, 4.13, 5.23,
                         6.26, 7.32, 4.90, 5.60, 6.44, 6.02, 4.33, 5.94, 6.21,
                         7.10, 4.69, 6.63, 6.43, 4.69],
    "Material Melting Rate": [100.71, 74.77, 119.88, 79.11, 93.27, 99.84,
                                   117.20, 95.14, 66.76, 82.97, 110.85, 115.82,
                                   90.99, 102.68, 115.36, 102.48, 86.10, 107.20,
                                   108.79, 122.52, 87.58, 114.14, 107.00, 78.57],
    "Heat Affected Zone": [90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89,
                           88.12, 74.46, 85.36, 93.12, 108.08, 82.88, 86.30,
                           93.28, 89.51, 75.64, 89.40, 90.81, 106.40, 78.76,
                           94.76, 94.59, 81.59]
}

df = pd.DataFrame(data)

# ============ Font configuration ============
font_name = 'DejaVu Sans'
font_size = 14
title_size = 16
font_weight = 'bold'

# ============ Features and targets ============
X = df[['Laser Power','Scanning Speed','Number of passes','Laser Frequency']]
targets = ['Surface Roughness', 'Material Melting Rate', 'Heat Affected Zone']

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
