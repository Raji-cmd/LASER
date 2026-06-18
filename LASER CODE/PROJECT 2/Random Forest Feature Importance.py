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

# === Features and Targets ===
X = df[['Laser Power','Scanning Speed','Number of passes','Laser Frequency']]

targets = {'Surface Roughness': df['Surface Roughness'],
    'Material Melting Rate': df['Material Melting Rate'],
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
