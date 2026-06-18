import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============ Input Parameters ============
inputs = {
    "PLZ": [8.5, 6.5, 10.5, 6.5, 6.5, 8.5, 10.5, 8.5, 6.5, 10.5, 8.5, 10.5,
            10.5, 6.5, 8.5, 7.5, 10.5, 8.5, 9.5, 6.5, 6.5, 6.5, 10.5, 10.5],
    "SSP": [280, 360, 200, 360, 200, 280, 200, 320, 360, 360, 280, 200,
            360, 200, 240, 280, 360, 280, 280, 200, 360, 200, 200, 360],
    "PC": [24, 12, 36, 36, 12, 18, 12, 24, 12, 36, 30, 36,
           36, 12, 24, 24, 12, 24, 24, 36, 36, 36, 12, 12],
    "FLZ": [48, 18, 18, 58, 58, 38, 18, 38, 58, 58, 38, 58,
            18, 18, 38, 38, 18, 28, 38, 58, 18, 18, 58, 58],
}

# ============ Output Parameters ============

outputs = {
    "SR": [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66, 4.13, 5.23, 6.26, 7.32,
           4.90, 5.60, 6.44, 6.02, 4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69],

    "MMR": [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14, 66.76,
            82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48, 86.10,
            107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57],

    "HAZ": [90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12, 74.46,
            85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51, 75.64,
            89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59]

}
# ============ DataFrame ============
df = pd.DataFrame({**inputs, **outputs})

# ============ Compute correlation matrix ============
corr_matrix = df.corr()

# ============ font style ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# ============ Plot heatmap ============
plt.figure(figsize=(10, 8))
ax=sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="cividis",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
# ---- X-axis labels to TOP ----
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
# ============ Title and Axis Label ============
plt.title("Correlation Matrix", fontsize=16, fontweight='bold', fontfamily='DejaVu Sans', pad=15)
plt.xticks(rotation=90, ha='right', fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
plt.yticks(rotation=0, fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
plt.tight_layout()
plt.show()
