import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ================= Dataset =================
data = {
    "Laser Power": [8.5, 6.5, 10.5, 6.5, 6.5, 8.5, 10.5, 8.5, 6.5, 10.5, 8.5, 10.5,
                    10.5, 6.5, 8.5, 7.5, 10.5, 8.5, 9.5, 6.5, 6.5, 6.5, 10.5, 10.5],
    "Scanning Speed": [280, 360, 200, 360, 200, 280, 200, 320, 360, 360, 280, 200,
                       360, 200, 240, 280, 360, 280, 280, 200, 360, 200, 200, 360],
    "Number of passes": [24, 12, 36, 36, 12, 18, 12, 24, 12, 36, 30, 36,
                          36, 12, 24, 24, 12, 24, 24, 36, 36, 36, 12, 12],
    "Laser Frequency": [48, 18, 18, 58, 58, 38, 18, 38, 58, 58, 38, 58,
                        18, 18, 38, 38, 18, 28, 38, 58, 18, 18, 58, 58],
     "External Surface Roughness": [6.054, 4.093, 6.795, 5.116, 6.067, 5.807, 6.068, 5.566, 4.259, 5.295, 6.179, 7.227, 4.984, 5.707
        , 6.323, 5.916, 4.394, 5.886, 6.072, 7.057, 4.788, 6.629, 6.412, 4.701],

    "Material Removal Quantity": [101.16, 77.37, 120.03, 80.90, 95.76, 99.76, 114.76, 95.06, 70.94, 85.12, 108.39, 116.43,
         91.69, 103.77, 112.34, 101.20, 85.50, 105.68, 106.52, 117.09, 87.33, 115.22, 107.58, 79.59],

    "Heat Affected Zone": [90.337, 72.734, 101.499, 84.786, 90.456, 87.866, 90.365, 87.044, 75.572, 86.161, 92.538, 105.676, 83.136,
        86.988, 93.153, 88.969, 76.033, 88.999, 90.520, 103.685, 79.428, 96.158, 94.727, 81.066]
}

df = pd.DataFrame(data)

# ================= Objectives =================
esr = df['External Surface Roughness'].values   # Minimize
haz = df['Heat Affected Zone'].values  # Minimize

# ================= Pareto Identification =================
def identify_pareto(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(scores[is_efficient] < c, axis=1)
            )
            is_efficient[i] = True
    return is_efficient

scores = np.column_stack([esr, haz])  # both minimized
pareto_mask = identify_pareto(scores)
pareto_points = np.column_stack([esr[pareto_mask], haz[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 0].argsort()]

# ================= Key Solutions =================
best_esr_idx = np.argmin(esr)
best_haz_idx = np.argmin(haz)
balanced_idx = np.argmin(
    np.sqrt((esr - esr.min())**2 + (haz - haz.min())**2)
)

# ================= Plot =================
plt.figure(figsize=(10, 8))

plt.scatter(esr, haz, color='gray', alpha=0.6, s=50, label='Experimental Points')

plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=90, edgecolors='black',
            label='Pareto-optimal Solutions')

plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.8)

key_points = [
    (esr[best_esr_idx], haz[best_esr_idx], "Min ESR", 's', 'red'),
    (esr[best_haz_idx], haz[best_haz_idx], "Min HAZ", '^', 'green'),
    (esr[balanced_idx], haz[balanced_idx], "Balanced", 'D', 'purple')
]

for x, y, label, marker, color in key_points:
    plt.scatter(x, y, s=150, marker=marker,
                color=color, edgecolors='black', linewidth=2)

    if label == "Balanced":
        offset = (-150, 10)
    else:
        offset = (25, -30)

    plt.annotate(f"{label}\nESR={x:.2f}, HAZ={y:.2f}",
                 xy=(x, y),
                 xytext=offset,
                 textcoords='offset points',
                 fontsize=10,
                 fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', lw=1.2))

if len(pareto_points) >= 3:
    hull = ConvexHull(pareto_points)
    plt.fill(pareto_points[hull.vertices, 0],
             pareto_points[hull.vertices, 1],
             alpha=0.12, label='Feasible Region')

plt.xlabel('External Surface Roughness (μm)', fontsize=14, fontweight='bold')
plt.ylabel('Heat Affected Zone (μm)', fontsize=14, fontweight='bold')
plt.title('Pareto Frontier for Laser Processing\nTrade-off between ESR and HAZ',
          fontsize=16, fontweight='bold', pad=18)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ================= Summary Table =================
best_esr = df.iloc[best_esr_idx]
best_haz = df.iloc[best_haz_idx]
balanced = df.iloc[balanced_idx]

summary_table = pd.DataFrame({
    'Parameter': [
        'Laser Power',
        'Scanning Speed',
        'Number of passes',
        'Laser Frequency',
        'External Surface Roughness (μm)',
        'Heat Affected Zone (μm)'
    ],
    'Min ESR': [
        best_esr['Laser Power'],
        best_esr['Scanning Speed'],
        best_esr['Number of passes'],
        best_esr['Laser Frequency'],
        best_esr['External Surface Roughness'],
        best_esr['Heat Affected Zone']
    ],
    'Min HAZ': [
        best_haz['Laser Power'],
        best_haz['Scanning Speed'],
        best_haz['Number of passes'],
        best_haz['Laser Frequency'],
        best_haz['External Surface Roughness'],
        best_haz['Heat Affected Zone']
    ],
    'Balanced': [
        balanced['Laser Power'],
        balanced['Scanning Speed'],
        balanced['Number of passes'],
        balanced['Laser Frequency'],
        balanced['External Surface Roughness'],
        balanced['Heat Affected Zone']
    ]
})

print("\n=========== Table: Optimal Process Parameters (SU–HAZ) ===========")
print(summary_table.to_string(index=False))
print("=================================================================")