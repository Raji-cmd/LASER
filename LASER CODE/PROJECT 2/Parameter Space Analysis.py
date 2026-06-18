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

    "Surface Roughness": [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66, 4.13, 5.23, 6.26, 7.32,
                         4.90, 5.60, 6.44, 6.02, 4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69],

    "Material Melting Rate": [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14, 66.76,
                                   82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48, 86.10,
                                   107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57],

    "Heat Affected Zone": [90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12, 74.46,
                           85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51, 75.64,
                           89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59]
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

# ============ Plot 1 — Surface Roughness ============
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(df['Laser Power'],
                     df['Scanning Speed'],
                     c=df['Surface Roughness'],
                     cmap='viridis',
                     s=100,
                     alpha=0.85,
                     edgecolors='black')

ax.set_xlabel('Laser Power', fontsize=14, fontweight='bold')
ax.set_ylabel('Scanning Speed', fontsize=14, fontweight='bold')
ax.set_title('Surface Roughness', fontsize=16, fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.3)
bold_axes(ax)

cbar = plt.colorbar(scatter)
cbar.set_label('Surface Roughness ', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ============ Plot 2 — MMR ============
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(df['Number of passes'],
                     df['Laser Frequency'],
                     c=df['Material Melting Rate'],
                     cmap='plasma',
                     s=100,
                     alpha=0.85,
                     edgecolors='black')

ax.set_xlabel('Number of passes', fontsize=14, fontweight='bold')
ax.set_ylabel('Laser Frequency', fontsize=14, fontweight='bold')
ax.set_title('MMR', fontsize=16, fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.3)
bold_axes(ax)

cbar = plt.colorbar(scatter)
cbar.set_label('MMR ', fontsize=12, fontweight='bold')

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
