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
    "Surface Waviness": [6.058, 4.045, 6.831, 5.075, 6.090, 5.837, 6.086, 5.595,4.216, 5.269, 6.179, 7.253, 4.954, 5.688, 6.289, 5.945,
    4.361, 5.908, 6.083, 7.097, 4.757, 6.648, 6.438, 4.668],
    "Material Vaporization Rate": [101.242, 75.847, 120.486, 79.510, 95.451, 100.088, 115.970, 95.406,
    68.498, 83.970, 108.874, 116.749, 91.344, 103.544, 112.447, 101.373,
    85.379, 105.910, 106.919, 119.898, 87.092, 115.320, 107.522, 78.744],
    "Heat Affected Zone": [90.275, 72.287, 102.351, 84.596, 90.363, 88.070, 90.342, 87.704,
    75.089, 85.898, 92.626, 107.341, 82.966, 86.991, 92.841, 88.993,
    75.926, 89.110, 90.525, 105.305, 79.328, 96.297, 94.604, 81.024]
}

df = pd.DataFrame(data)

# ================= Objectives =================
sw = df['Surface Waviness'].values              # Minimize
mvr = df['Material Vaporization Rate'].values   # Maximize

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

scores = np.column_stack([sw, -mvr])
pareto_mask = identify_pareto(scores)
pareto_points = np.column_stack([sw[pareto_mask], mvr[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 1].argsort()]

# ================= Key Solutions =================
best_quality_idx = np.argmin(sw)
best_productivity_idx = np.argmax(mvr)
balanced_idx = np.argmin(np.sqrt((sw - sw.min())**2 + (mvr - mvr.max())**2))

# ================= Plot =================
plt.figure(figsize=(10, 8))
plt.scatter(sw, mvr, color='gray', alpha=0.6, s=50, label='Experimental Points')

plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=80, edgecolors='black',
            label='Pareto-optimal Solutions')

plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.5)

key_points = [
    (best_quality_idx, 'Best Quality (Min SW)', 's', 'red'),
    (best_productivity_idx, 'Best Productivity (Max MVR)', '^', 'green'),
    (balanced_idx, 'Balanced Solution', 'D', 'purple')
]

for idx, label, marker, color in key_points:
    x, y = sw[idx], mvr[idx]
    plt.scatter(x, y, s=150, marker=marker,
                color=color, edgecolors='black', linewidth=2)

    # Move only Balanced label away to avoid overlap
    if "Balanced" in label:
        offset = (-140, -10)   # move left
    else:
        offset = (20, -40)     # default

    plt.annotate(f"{label}\nSW={x:.2f}, MVR={y:.2f}",
                 xy=(x, y),
                 xytext=offset,
                 textcoords='offset points',
                 fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', lw=1.2))


if len(pareto_points) >= 3:
    hull = ConvexHull(pareto_points)
    plt.fill(pareto_points[hull.vertices, 0],
             pareto_points[hull.vertices, 1],
             alpha=0.12, label='Feasible Region')

plt.xlabel('Surface Waviness  ', fontsize=14, fontweight='bold')
plt.ylabel('Material Vaporization Rate ', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.title('Pareto Frontier for Laser Processing\nTrade-off between SW and MVR',
          fontsize=16, fontweight='bold', pad=20)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ================= Summary =================
print(f"Total Pareto-optimal solutions: {pareto_mask.sum()}")
print(f"Best Quality → SW={sw[best_quality_idx]:.2f}, MVR={mvr[best_quality_idx]:.2f}")
print(f"Best Productivity → SW={sw[best_productivity_idx]:.2f}, MVR={mvr[best_productivity_idx]:.2f}")
print(f"Balanced → SW={sw[balanced_idx]:.2f}, MVR={mvr[balanced_idx]:.2f}")

best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
balanced = df.iloc[balanced_idx]

summary_table = pd.DataFrame({
    'Parameter': [
        'Laser Power',
        'Scanning Speed',
        'Number of Passes',
        'Laser Frequency',
        'Surface Waviness (μm)',
        'Material Vaporization Rate (mm³/min)',
        'Heat Affected Zone (μm)'
    ],
    'Best Quality (Min SW)': best_quality.values,
    'Best Productivity (Max MVR)': best_productivity.values,
    'Balanced Solution': balanced.values
})

print("\n=========== Table 3: Optimal Process Parameters ===========")
print(summary_table.to_string(index=False))
print("==========================================================")
