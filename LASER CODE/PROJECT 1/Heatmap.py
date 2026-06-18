import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============ Dataset ============
surface = pd.DataFrame({
    'Actual': [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
    4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
    4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69],
    'Predicted': [6.058, 4.045, 6.831, 5.075, 6.090, 5.837, 6.086, 5.595,
    4.216, 5.269, 6.179, 7.253, 4.954, 5.688, 6.289, 5.945,
    4.361, 5.908, 6.083, 7.097, 4.757, 6.648, 6.438, 4.668]
})

mvr = pd.DataFrame({
    'Actual': [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
    66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
    86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57],
    'Predicted': [101.242, 75.847, 120.486, 79.510, 95.451, 100.088, 115.970, 95.406,
    68.498, 83.970, 108.874, 116.749, 91.344, 103.544, 112.447, 101.373,
    85.379, 105.910, 106.919, 119.898, 87.092, 115.320, 107.522, 78.744]

})
HAZ = pd.DataFrame({
    'Actual': [ 90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
    74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
    75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59],
    'Predicted': [90.275, 72.287, 102.351, 84.596, 90.363, 88.070, 90.342, 87.704,
    75.089, 85.898, 92.626, 107.341, 82.966, 86.991, 92.841, 88.993,
    75.926, 89.110, 90.525, 105.305, 79.328, 96.297, 94.604, 81.024]
})
# ============ Output Parameter ============
datasets = {
    "Surface Waviness ": surface,
    "MVR": mvr,
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
        cmap='Blues',
        interpolation='nearest'
    )

# ============ Title and Axis Label ============
    plt.title(f'{name}: Actual vs Predicted Heat Map', fontsize=16,fontweight='bold',fontfamily='DejaVu Sans')
    plt.xlabel('Sample Index',fontsize=14,fontweight='bold',fontfamily='DejaVu Sans')
    plt.yticks([0, 1], ['Actual', 'Predicted'],fontsize=14,fontweight='bold',fontfamily='DejaVu Sans' )
    plt.xticks(fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')
    plt.yticks(fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')

# ============ Colorbar with Label ============
    cbar = plt.colorbar(im)
    cbar.set_label(
        'Value',
        fontsize=18,
        fontweight='bold',
        fontfamily='DejaVu Sans'
    )
    cbar.ax.tick_params(labelsize=14)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('DejaVu Sans')

# ============ Style & Layout ============
    plt.grid(False)
    plt.tight_layout()
    plt.show()
