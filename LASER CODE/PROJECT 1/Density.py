import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============ DataSet ============
surface_data = {
    'Exp.value': [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
                  4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69],
    'Ensemble': [6.058, 4.045, 6.831, 5.075, 6.090, 5.837, 6.086, 5.595,4.216, 5.269, 6.179, 7.253, 4.954, 5.688, 6.289, 5.945,
    4.361, 5.908, 6.083, 7.097, 4.757, 6.648, 6.438, 4.668]
}

mvr_data = {
    'Exp.value': [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
               66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
               86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57],
    'Ensemble': [ 101.242, 75.847, 120.486, 79.510, 95.451, 100.088, 115.970, 95.406,
    68.498, 83.970, 108.874, 116.749, 91.344, 103.544, 112.447, 101.373,
    85.379, 105.910, 106.919, 119.898, 87.092, 115.320, 107.522, 78.744]

}
haz_data = {
    'Exp.value': [90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
    74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
    75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59],
    'Ensemble': [90.275, 72.287, 102.351, 84.596, 90.363, 88.070, 90.342, 87.704,
    75.089, 85.898, 92.626, 107.341, 82.966, 86.991, 92.841, 88.993,
    75.926, 89.110, 90.525, 105.305, 79.328, 96.297, 94.604, 81.024
]
}

# ============ DataFrames ============
datasets = {
    "Surface waviness": pd.DataFrame(surface_data),
    "MVR": pd.DataFrame(mvr_data),
    "HAZ": pd.DataFrame(haz_data)
}
# ============ Font Settings ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# ============ Color scheme ============

colors = ['#1f77b4', '#ff7f0e']  # Blue: Actual, Orange: Predicted

# ============ Plot Density & Summary ============
def plot_and_summarize(df, metric_name):
    plt.figure(figsize=(6,4))
    sns.kdeplot(df['Exp.value'], label='Actual', color=colors[0], linewidth=2, fill=True, alpha=0.5, bw_adjust=0.5)
    sns.kdeplot(df['Ensemble'], label='Predicted', color=colors[1], linewidth=2, fill=True, alpha=0.5, bw_adjust=0.5)

# ============ Title and Axis Label ============
    plt.title(f"{metric_name}: Actual vs Predicted Density",fontsize=16, fontweight='bold', fontname='DejaVu Sans')
    plt.xlabel(metric_name, fontsize=14, fontweight='bold', fontname='DejaVu Sans')
    plt.ylabel('Density', fontsize=14, fontweight='bold', fontname='DejaVu Sans')
    plt.xticks(fontsize=12, fontweight='bold', fontname='DejaVu Sans')
    plt.yticks(fontsize=12, fontweight='bold', fontname='DejaVu Sans')
    plt.legend(fontsize=10, title_fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============ Statistical Summary ============
    print(f"\n=== {metric_name} Statistical Summary ===")
    print(f"Actual   - Mean: {df['Exp.value'].mean():.4f}, Std: {df['Exp.value'].std():.4f}")
    print(f"Predicted- Mean: {df['Ensemble'].mean():.4f}, Std: {df['Ensemble'].std():.4f}")

    ks_stat, p_value = stats.ks_2samp(df['Exp.value'], df['Ensemble'])
    print(f"Kolmogorov-Smirnov Test: Stat={ks_stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print("Distributions are statistically similar (p > 0.05)")
    else:
        print("Distributions are statistically different (p ≤ 0.05)")

# ============ Apply Function to All Datasets ============
for name, df in datasets.items():
    plot_and_summarize(df, name)
