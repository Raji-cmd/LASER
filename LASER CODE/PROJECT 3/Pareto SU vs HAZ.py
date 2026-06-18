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
    "Surface Undulation": [5.979, 4.191, 6.791, 5.173, 6.087, 5.733, 6.086, 5.498,
                           4.384, 5.35, 6.1, 7.18, 5.054, 5.82, 6.292, 5.835,
                           4.45, 5.809, 5.988, 6.991, 4.856, 6.644, 6.395, 4.731],
    "Material Removal Rate": [100.401, 79.131, 120.923, 82.47, 98.176, 98.832, 113.374, 94.31,
                              73.706, 87.317, 106.071, 117.261, 92.83, 105.097, 110.057, 100.027,
                              85.243, 104.087, 104.521, 113.161, 87.554, 116.303, 108.177, 80.311],
    "Heat Affected Zone": [90.228, 73.02, 100.662, 84.913, 91.161, 87.43, 90.805, 86.026,
                           76.361, 87.244, 91.945, 104.26, 83.62, 87.424, 93.065, 88.651,
                           76.144, 88.5, 90.263, 101.782, 80.061, 96.977, 95.142, 80.711]
}

df = pd.DataFrame(data)

# ================= Objectives =================
su = df['Surface Undulation'].values   # Minimize
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

scores = np.column_stack([su, haz])  # both minimized
pareto_mask = identify_pareto(scores)
pareto_points = np.column_stack([su[pareto_mask], haz[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 0].argsort()]

# ================= Key Solutions =================
best_su_idx = np.argmin(su)
best_haz_idx = np.argmin(haz)
balanced_idx = np.argmin(
    np.sqrt((su - su.min())**2 + (haz - haz.min())**2)
)

# ================= Plot =================
plt.figure(figsize=(10, 8))

plt.scatter(su, haz, color='gray', alpha=0.6, s=50, label='Experimental Points')

plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=90, edgecolors='black',
            label='Pareto-optimal Solutions')

plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.8)

key_points = [
    (su[best_su_idx], haz[best_su_idx], "Min SU", 's', 'red'),
    (su[best_haz_idx], haz[best_haz_idx], "Min HAZ", '^', 'green'),
    (su[balanced_idx], haz[balanced_idx], "Balanced", 'D', 'purple')
]

for x, y, label, marker, color in key_points:
    plt.scatter(x, y, s=150, marker=marker,
                color=color, edgecolors='black', linewidth=2)

    if label == "Balanced":
        offset = (-150, 10)
    else:
        offset = (25, -30)

    plt.annotate(f"{label}\nSU={x:.2f}, HAZ={y:.2f}",
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

plt.xlabel('Surface Undulation (μm)', fontsize=14, fontweight='bold')
plt.ylabel('Heat Affected Zone (μm)', fontsize=14, fontweight='bold')
plt.title('Pareto Frontier for Laser Processing\nTrade-off between SU and HAZ',
          fontsize=16, fontweight='bold', pad=18)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ================= Summary Table =================
best_su = df.iloc[best_su_idx]
best_haz = df.iloc[best_haz_idx]
balanced = df.iloc[balanced_idx]

summary_table = pd.DataFrame({
    'Parameter': [
        'Laser Power',
        'Scanning Speed',
        'Number of passes',
        'Laser Frequency',
        'Surface Undulation (μm)',
        'Heat Affected Zone (μm)'
    ],
    'Min SU': [
        best_su['Laser Power'],
        best_su['Scanning Speed'],
        best_su['Number of passes'],
        best_su['Laser Frequency'],
        best_su['Surface Undulation'],
        best_su['Heat Affected Zone']
    ],
    'Min HAZ': [
        best_haz['Laser Power'],
        best_haz['Scanning Speed'],
        best_haz['Number of passes'],
        best_haz['Laser Frequency'],
        best_haz['Surface Undulation'],
        best_haz['Heat Affected Zone']
    ],
    'Balanced': [
        balanced['Laser Power'],
        balanced['Scanning Speed'],
        balanced['Number of passes'],
        balanced['Laser Frequency'],
        balanced['Surface Undulation'],
        balanced['Heat Affected Zone']
    ]
})

print("\n=========== Table: Optimal Process Parameters (SU–HAZ) ===========")
print(summary_table.to_string(index=False))
print("=================================================================")