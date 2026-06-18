import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# === Dataset ===
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

# === Features and Targets ===
X = df[['Laser Power','Scanning Speed','Number of passes','Laser Frequency']]

targets = {'External Surface Roughness': df['External Surface Roughness'],
    'Material Removal Quantity': df['Material Removal Quantity'],
    'Heat Affected Zone': df['Heat Affected Zone']
}

# === Train models and plot individually ===
for name, y in targets.items():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=importance.values,
        y=importance.index,
        palette="viridis"
    )

# ============ Title and Axis Label ============
    plt.title(f"Random Forest Feature Importance\n{name}",fontweight='bold', fontsize=16, pad=15)
    plt.xlabel("Feature Importance Score", fontsize=14, fontweight='bold')
    plt.ylabel("Features", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
