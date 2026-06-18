import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============ Dataset ============

external_surface_roughness = pd.DataFrame({
    'Actual': [
        6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
        4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
        4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69
    ],
    'Predicted': [
        5.979, 4.191, 6.791, 5.173, 6.087, 5.733, 6.086, 5.498,
        4.384, 5.35, 6.1, 7.18, 5.054, 5.82, 6.292, 5.835,
        4.45, 5.809, 5.988, 6.991, 4.856, 6.644, 6.395, 4.731
    ]
})

material_removal_quantity = pd.DataFrame({
    'Actual': [
        100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
        66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
        86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57
    ],
    'Predicted': [
        101.16, 77.37, 120.03, 80.90, 95.76, 99.76, 114.76, 95.06, 70.94, 85.12, 108.39, 116.43,
        91.69, 103.77, 112.34, 101.20, 85.50, 105.68, 106.52, 117.09, 87.33, 115.22, 107.58, 79.59,
    ]
})

HAZ = pd.DataFrame({
    'Actual': [
        90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
        74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
        75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59
    ],
    'Predicted': [
        90.337, 72.734, 101.499, 84.786, 90.456, 87.866, 90.365, 87.044, 75.572, 86.161, 92.538, 105.676, 83.136,
        86.988, 93.153, 88.969, 76.033, 88.999, 90.520, 103.685, 79.428, 96.158, 94.727, 81.066,
    ]
})

# ============ Output Parameter ============
datasets = {
    "External Surface Roughness": external_surface_roughness,
    "Material Removal Quantity": material_removal_quantity,
    "HAZ": HAZ,
}

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# ============ Heatmap Plot ============
for name, df in datasets.items():
    plt.figure(figsize=(12, 4))
    heat_matrix = np.vstack([df['Actual'].values, df['Predicted'].values])
    im = plt.imshow(
        heat_matrix,
        aspect='auto',
        cmap='Greens',
        interpolation='nearest'
    )

    # ============ Title and Axis Label ============
    plt.title(f'{name}: Actual vs Predicted Heat Map', fontsize=16, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=14, fontweight='bold')
    plt.yticks([0, 1], ['Actual', 'Predicted'], fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')

    # ============ Colorbar with Label ============
    cbar = plt.colorbar(im)
    cbar.set_label('Value', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    # ============ Style & Layout ============
    plt.grid(False)
    plt.tight_layout()
    plt.show()