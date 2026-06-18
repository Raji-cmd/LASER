import pandas as pd
import matplotlib.pyplot as plt

# ============ Dataset ============
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

# Global font
plt.rcParams['font.family'] = 'DejaVu Sans'

def bold_axes(ax):
    ax.tick_params(axis='both', labelsize=12, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

# ============ Plot 1 — Surface Undulation ============
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(df['Laser Power'],
                     df['Scanning Speed'],
                     c=df['Surface Undulation'],
                     cmap='viridis',
                     s=100,
                     alpha=0.85,
                     edgecolors='black')

ax.set_xlabel('Laser Power', fontsize=14, fontweight='bold')
ax.set_ylabel('Scanning Speed', fontsize=14, fontweight='bold')
ax.set_title('Surface Undulation', fontsize=16, fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.3)
bold_axes(ax)

cbar = plt.colorbar(scatter)
cbar.set_label('Surface Undulation ', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ============ Plot 2 — MRR ============
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(df['Number of passes'],
                     df['Laser Frequency'],
                     c=df['Material Removal Rate'],
                     cmap='plasma',
                     s=100,
                     alpha=0.85,
                     edgecolors='black')

ax.set_xlabel('Number of passes', fontsize=14, fontweight='bold')
ax.set_ylabel('Laser Frequency', fontsize=14, fontweight='bold')
ax.set_title('MRR', fontsize=16, fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.3)
bold_axes(ax)

cbar = plt.colorbar(scatter)
cbar.set_label('MRR ', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ============ Plot 3 — HAZ ============
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(df['Laser Power'],
                     df['Laser Frequency'],
                     c=df['Heat Affected Zone'],
                     cmap='cool',
                     s=100,
                     alpha=0.85,
                     edgecolors='black')

ax.set_xlabel('Laser Power', fontsize=14, fontweight='bold')
ax.set_ylabel('Laser Frequency', fontsize=14, fontweight='bold')
ax.set_title('HAZ', fontsize=16, fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.3)
bold_axes(ax)

cbar = plt.colorbar(scatter)
cbar.set_label('HAZ', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

