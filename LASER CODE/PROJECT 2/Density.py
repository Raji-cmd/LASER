import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============ DataSet ============
surface_data = {
    'Exp.value': [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
                  4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69],
    'Ensemble': [6.039,4.108,6.794,5.14,6.103,5.774,6.08,5.524,4.274,
                    5.312,6.165,7.218,5.01,5.695,6.315,5.9,4.406,5.876,6.048,7.055,
                    4.819,6.608,6.429,4.728]
}

mmr_data = {
    'Exp.value': [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
               66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
               86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57],
    'Ensemble': [ 100.621, 76.813, 120.794, 81.028, 95.547, 99.342, 115.681,94.356, 70.089, 85.358,
                      108.255, 116.464, 92.321, 104.011, 112.836, 101.002, 85.953, 105.430,
                      106.462, 118.157, 88.009, 115.102, 107.506, 79.420]

}
haz_data = {
    'Exp.value': [90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
    74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
    75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59],
    'Ensemble': [90.394, 72.297, 101.676, 85.307, 90.512, 87.663, 90.311, 86.879, 75.356, 86.501,
                     92.662, 106.236, 83.418, 86.723, 93.389, 88.886,75.768,88.963,90.527,
                     104.153,79.393,95.349,95.011,81.308
]
}

# ============ DataFrames ============
datasets = {
    "Surface Roughness": pd.DataFrame(surface_data),
    "MMR": pd.DataFrame(mmr_data),
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
