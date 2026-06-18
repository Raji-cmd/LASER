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
    "External Surface Roughness": [6.054, 4.093, 6.795, 5.116, 6.067, 5.807, 6.068, 5.566, 4.259, 5.295, 6.179, 7.227,
                                   4.984, 5.707
        , 6.323, 5.916, 4.394, 5.886, 6.072, 7.057, 4.788, 6.629, 6.412, 4.701],

    "Material Removal Quantity": [101.16, 77.37, 120.03, 80.90, 95.76, 99.76, 114.76, 95.06, 70.94, 85.12, 108.39,
                                  116.43,
                                  91.69, 103.77, 112.34, 101.20, 85.50, 105.68, 106.52, 117.09, 87.33, 115.22, 107.58,
                                  79.59],
}

df = pd.DataFrame(data)

# ================= Objectives =================
esr = df['External Surface Roughness'].values          # Minimize
mrq = df['Material Removal Quantity'].values      # Maximize

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

scores = np.column_stack([esr, -mrq])
pareto_mask = identify_pareto(scores)
pareto_points = np.column_stack([esr[pareto_mask], mrq[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 1].argsort()]

# ================= Key Solutions =================
best_quality_idx = np.argmin(esr)
best_productivity_idx = np.argmax(mrq)
balanced_idx = np.argmin(np.sqrt((esr - esr.min())**2 + (mrq- mrq.max())**2))

# ================= Plot =================
plt.figure(figsize=(10, 8))
plt.scatter(esr, mrq, color='gray', alpha=0.6, s=50, label='Experimental Points')

plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=80, edgecolors='black',
            label='Pareto-optimal Solutions')

plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.5)

key_points = [
    (best_quality_idx, 'Best Quality (Min ESR)', 's', 'red'),
    (best_productivity_idx, 'Best Productivity (Max MRQ)', '^', 'green'),
    (balanced_idx, 'Balanced Solution', 'D', 'purple')
]

for idx, label, marker, color in key_points:
    x, y = esr[idx], mrq[idx]
    plt.scatter(x, y, s=150, marker=marker,
                color=color, edgecolors='black', linewidth=2)

    # Move only Balanced label away to avoid overlap
    if "Balanced" in label:
        offset = (-140, -10)
    else:
        offset = (20, -40)

    plt.annotate(f"{label}\nESR={x:.2f}, MRQ={y:.2f}",
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

plt.xlabel('External Surface Roughness (μm)', fontsize=14, fontweight='bold')
plt.ylabel('Material Removal Quantity(mm³/min)', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.title('Pareto Frontier for Laser Processing\nTrade-off between ESR and MRQ',
          fontsize=16, fontweight='bold', pad=20)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ================= Summary =================
print(f"Total Pareto-optimal solutions: {pareto_mask.sum()}")
print(f"Best Quality → ESR={esr[best_quality_idx]:.2f}, MRQ={mrq[best_quality_idx]:.2f}")
print(f"Best Productivity → ESR={esr[best_productivity_idx]:.2f}, MRQ={mrq[best_productivity_idx]:.2f}")
print(f"Balanced → ESR={esr[balanced_idx]:.2f}, MRQ={mrq[balanced_idx]:.2f}")

best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
balanced = df.iloc[balanced_idx]

summary_table = pd.DataFrame({
    'Parameter': [
        'Laser Power',
        'Scanning Speed',
        'Number of Passes',
        'Laser Frequency',
        'External Surface Roughness (μm)',
        'Material Removal Quantity (mm³/min)',
    ],
    'Best Quality (Min ESR)': best_quality.values,
    'Best Productivity (Max MRQ)': best_productivity.values,
    'Balanced Solution': balanced.values
})

print("\n=========== Table: Optimal Process Parameters (ESR–MRQ) ===========")
print(summary_table.to_string(index=False))
print("==========================================================")