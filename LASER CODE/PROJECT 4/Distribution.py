import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============ DataSet ============
data = {
    "PLZ": [8.5, 6.5, 10.5, 6.5, 6.5, 8.5, 10.5, 8.5, 6.5, 10.5, 8.5, 10.5,
            10.5, 6.5, 8.5, 7.5, 10.5, 8.5, 9.5, 6.5, 6.5, 6.5, 10.5, 10.5],
    "SSP": [280, 360, 200, 360, 200, 280, 200, 320, 360, 360, 280, 200,
            360, 200, 240, 280, 360, 280, 280, 200, 360, 200, 200, 360],
    "PC": [24, 12, 36, 36, 12, 18, 12, 24, 12, 36, 30, 36,
           36, 12, 24, 24, 12, 24, 24, 36, 36, 36, 12, 12],
    "FLZ": [48, 18, 18, 58, 58, 38, 18, 38, 58, 58, 38, 58,
            18, 18, 38, 38, 18, 28, 38, 58, 18, 18, 58, 58],

    "External Surface Roughness": [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66, 4.13, 5.23, 6.26, 7.32,
           4.90, 5.60, 6.44, 6.02, 4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69],

    "MRQ": [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14, 66.76,
            82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48, 86.10,
            107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57],

    "HAZ": [90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12, 74.46,
            85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51, 75.64,
            89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59]
}

df = pd.DataFrame(data)

# ============ Train/Test split ============
targets = ["External Surface Roughness", "MRQ", "HAZ"]
train_test_data = {}

for target in targets:
    train, test = train_test_split(df[target], test_size=0.2, random_state=42)
    train_test_data[target] = (train, test)

# ============ font style ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

colors = {
    'External Surface Roughness': ('skyblue', 'royalblue'),
    'MRQ': ('lightgreen','darkgreen'),
    'HAZ': ('pink', 'brown')
}

# ============ Separate Plot for Each Target ============
for target in targets:
    train, test = train_test_data[target]

    plt.figure(figsize=(7,5))   # NEW figure for each output

    plt.hist(train, bins=8, alpha=0.7,
             color=colors[target][0],
             edgecolor='black',
             label='Train')

    plt.hist(test, bins=8, alpha=0.7,
             color=colors[target][1],
             edgecolor='black',
             label='Test')

    plt.title(f'{target} Distribution',
              fontsize=16, fontweight='bold')
    plt.xlabel(target,
               fontsize=14, fontweight='bold')
    plt.ylabel('Frequency',
               fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

