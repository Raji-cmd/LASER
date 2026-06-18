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
    "Surface Roughness": [6.039, 4.108, 6.794, 5.14, 6.103, 5.774, 6.08, 5.524, 4.274,
                  5.312, 6.165, 7.218, 5.01, 5.695, 6.315, 5.9, 4.406, 5.876,
                  6.048, 7.055, 4.819, 6.608, 6.429, 4.728],
    "Material Melting Rate": [100.621, 76.813, 120.794, 81.028, 95.547, 99.342, 115.681,
                  94.356, 70.089, 85.358, 108.255, 116.464, 92.321, 104.011,
                  112.836, 101.002, 85.953, 105.430, 106.462, 118.157,
                  88.009, 115.102, 107.506, 79.420],
    "Heat Affected Zone": [90.394, 72.297, 101.676, 85.307, 90.512, 87.663, 90.311,
                  86.879, 75.356, 86.501, 92.662, 106.236, 83.418, 86.723,
                  93.389, 88.886, 75.768, 88.963, 90.527, 104.153,
                  79.393, 95.349, 95.011, 81.308]
}

df = pd.DataFrame(data)

# ================= Objectives =================
haz = df['Heat Affected Zone'].values          # Minimize
mmr = df['Material Melting Rate'].values  # Maximize

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

scores = np.column_stack([haz, -mmr])
pareto_mask = identify_pareto(scores)
pareto_points = np.column_stack([haz[pareto_mask], mmr[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 1].argsort()]

# ================= Key Solutions =================
best_quality_idx = np.argmin(haz)
best_productivity_idx = np.argmax(mmr)
balanced_idx = np.argmin(np.sqrt((haz - haz.min())**2 + (mmr - mmr.max())**2))

# ================= Plot =================
plt.figure(figsize=(10, 8))

plt.scatter(haz, mmr, color='gray', alpha=0.6, s=50, label='Experimental Points')

plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=90, edgecolors='black',
            label='Pareto-optimal Solutions')

plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.8)

key_points = [
    (haz[best_quality_idx], mmr[best_quality_idx], "Min HAZ", 's', 'red'),
    (haz[best_productivity_idx], mmr[best_productivity_idx], "Max MMR", '^', 'green'),
    (haz[balanced_idx], mmr[balanced_idx], "Balanced", 'D', 'purple')
]

for x, y, label, marker, color in key_points:
    plt.scatter(x, y, s=150, marker=marker,
                color=color, edgecolors='black', linewidth=2)

    # Shift balanced annotation to avoid overlap
    if label == "Balanced":
        offset = (-150, 10)
    else:
        offset = (25, -30)

    plt.annotate(f"{label}\nHAZ={x:.2f}, MMR={y:.2f}",
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

plt.xlabel('Heat Affected Zone ', fontsize=14, fontweight='bold')
plt.ylabel('Material Melting Rate', fontsize=14, fontweight='bold')
plt.title('Pareto Frontier for Laser Processing\nTrade-off between HAZ and MMR',
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
        'Material Melting Rate'
    ],
    'Min HAZ': [
        best_quality['Laser Power'],
        best_quality['Scanning Speed'],
        best_quality['Number of passes'],
        best_quality['Laser Frequency'],
        best_quality['Heat Affected Zone'],
        best_quality['Material Melting Rate']
    ],
    'Max MVR': [
        best_productivity['Laser Power'],
        best_productivity['Scanning Speed'],
        best_productivity['Number of passes'],
        best_productivity['Laser Frequency'],
        best_productivity['Heat Affected Zone'],
        best_productivity['Material Melting Rate']
    ],
    'Balanced': [
        balanced['Laser Power'],
        balanced['Scanning Speed'],
        balanced['Number of passes'],
        balanced['Laser Frequency'],
        balanced['Heat Affected Zone'],
        balanced['Material Melting Rate']
    ]
})

print("\n=========== Table 3: Optimal Process Parameters ===========")
print(summary_table.to_string(index=False))
print("==========================================================")
