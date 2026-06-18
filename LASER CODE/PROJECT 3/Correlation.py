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
    "ESR": [5.979, 4.191, 6.791, 5.173, 6.087, 5.733, 6.086, 5.498,
        4.384, 5.35, 6.1, 7.18, 5.054, 5.82, 6.292, 5.835,
        4.45, 5.809, 5.988, 6.991, 4.856, 6.644, 6.395, 4.731],

    "MRQ": [ 100.401, 79.131, 120.923, 82.47, 98.176, 98.832, 113.374, 94.31,
        73.706, 87.317, 106.071, 117.261, 92.83, 105.097, 110.057, 100.027,
        85.243, 104.087, 104.521, 113.161, 87.554, 116.303, 108.177, 80.311],

    "HAZ": [ 90.228, 73.02, 100.662, 84.913, 91.161, 87.43, 90.805, 86.026,
        76.361, 87.244, 91.945, 104.26, 83.62, 87.424, 93.065, 88.651,
        76.144, 88.5, 90.263, 101.782, 80.061, 96.977, 95.142, 80.711]

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
    cmap="RdBu_r",
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
