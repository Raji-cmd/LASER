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
    "Surface Undulation": [6.058, 4.045, 6.831, 5.075, 6.090, 5.837, 6.086, 5.595, 4.216, 5.269, 6.179, 7.253, 4.954,
                         5.688, 6.289, 5.945,
                         4.361, 5.908, 6.083, 7.097, 4.757, 6.648, 6.438, 4.668],
    "Material Removal Rate": [101.242, 75.847, 120.486, 79.510, 95.451, 100.088, 115.970, 95.406,
                                   68.498, 83.970, 108.874, 116.749, 91.344, 103.544, 112.447, 101.373,
                                   85.379, 105.910, 106.919, 119.898, 87.092, 115.320, 107.522, 78.744],
    "Heat Affected Zone": [90.275, 72.287, 102.351, 84.596, 90.363, 88.070, 90.342, 87.704,
                           75.089, 85.898, 92.626, 107.341, 82.966, 86.991, 92.841, 88.993,
                           75.926, 89.110, 90.525, 105.305, 79.328, 96.297, 94.604, 81.024]
}

df = pd.DataFrame(data)

# ================= Objectives =================
haz = df['Heat Affected Zone'].values          # Minimize
mrr = df['Material Removal Rate'].values       # Maximize

# ================= Pareto Identification =================
def identify_pareto(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(scores[is_efficient] < c, axis=1) |
                np.any(scores[is_efficient] > c, axis=1)
            )
            is_efficient[i] = True
    return is_efficient

scores = np.column_stack([haz, -mrr])
pareto_mask = identify_pareto(scores)
pareto_points = np.column_stack([haz[pareto_mask], mrr[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 1].argsort()]

# ================= Key Solutions =================
best_quality_idx = np.argmin(haz)
best_productivity_idx = np.argmax(mrr)
balanced_idx = np.argmin(np.sqrt((haz - haz.min())**2 + (mrr - mrr.max())**2))

# ================= Plot =================
plt.figure(figsize=(10, 8))

plt.scatter(haz, mrr, color='gray', alpha=0.6, s=50, label='Experimental Points')

plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=90, edgecolors='black',
            label='Pareto-optimal Solutions')

plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.8)

key_points = [
    (haz[best_quality_idx], mrr[best_quality_idx], "Min HAZ", 's', 'red'),
    (haz[best_productivity_idx], mrr[best_productivity_idx], "Max MRR", '^', 'green'),
    (haz[balanced_idx], mrr[balanced_idx], "Balanced", 'D', 'purple')
]

for x, y, label, marker, color in key_points:
    plt.scatter(x, y, s=150, marker=marker,
                color=color, edgecolors='black', linewidth=2)

    if label == "Balanced":
        offset = (-150, 10)
    else:
        offset = (25, -30)

    plt.annotate(f"{label}\nHAZ={x:.2f}, MRR={y:.2f}",
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

plt.xlabel('Heat Affected Zone', fontsize=14, fontweight='bold')
plt.ylabel('Material Removal Rate', fontsize=14, fontweight='bold')
plt.title('Pareto Frontier for Laser Processing\nTrade-off between HAZ and MRR',
          fontsize=16, fontweight='bold', pad=18)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ================= Summary Table =================
best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
balanced = df.iloc[balanced_idx]

summary_table = pd.DataFrame({
    'Parameter': [
        'Laser Power',
        'Scanning Speed',
        'Number of passes',
        'Laser Frequency',
        'Heat Affected Zone',
        'Material Removal Rate'
    ],
    'Min HAZ': [
        best_quality['Laser Power'],
        best_quality['Scanning Speed'],
        best_quality['Number of passes'],
        best_quality['Laser Frequency'],
        best_quality['Heat Affected Zone'],
        best_quality['Material Removal Rate']
    ],
    'Max MRR': [
        best_productivity['Laser Power'],
        best_productivity['Scanning Speed'],
        best_productivity['Number of passes'],
        best_productivity['Laser Frequency'],
        best_productivity['Heat Affected Zone'],
        best_productivity['Material Removal Rate']
    ],
    'Balanced': [
        balanced['Laser Power'],
        balanced['Scanning Speed'],
        balanced['Number of passes'],
        balanced['Laser Frequency'],
        balanced['Heat Affected Zone'],
        balanced['Material Removal Rate']
    ]
})

print("\n=========== Table 3: Optimal Process Parameters ===========")
print(summary_table.to_string(index=False))
print("==========================================================")