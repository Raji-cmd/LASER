import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# ---- Surface Waviness (SW) ----
sw_actual = np.array([
    6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
    4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
    4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69
])

# ---- Material Vaporization Rate (MVR) ----
mvr_actual = np.array([
    100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
    66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
    86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57
])

# ---- Heat Affected Zone (HAZ) ----
haz_actual = np.array([
    90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
    74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
    75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59
])

# ============ Predicted (Ensemble) values ============

sw_pred = np.array([
    6.058, 4.045, 6.831, 5.075, 6.090, 5.837, 6.086, 5.595,
    4.216, 5.269, 6.179, 7.253, 4.954, 5.688, 6.289, 5.945,
    4.361, 5.908, 6.083, 7.097, 4.757, 6.648, 6.438, 4.668
])

mvr_pred = np.array([
    101.242, 75.847, 120.486, 79.510, 95.451, 100.088, 115.970, 95.406,
    68.498, 83.970, 108.874, 116.749, 91.344, 103.544, 112.447, 101.373,
    85.379, 105.910, 106.919, 119.898, 87.092, 115.320, 107.522, 78.744
])

haz_pred = np.array([
    90.275, 72.287, 102.351, 84.596, 90.363, 88.070, 90.342, 87.704,
    75.089, 85.898, 92.626, 107.341, 82.966, 86.991, 92.841, 88.993,
    75.926, 89.110, 90.525, 105.305, 79.328, 96.297, 94.604, 81.024
])

# ============ Ensemble ============
datasets = {
    "Surface Waviness": (sw_actual + sw_pred) / 2,
    "MVR": (mvr_actual + mvr_pred) / 2,
    "HAZ":(haz_actual + haz_pred) /2,
}
# ============ sequence index ============
num_samples = len(sw_actual)
X = np.arange(num_samples).reshape(-1, 1)

# ============ Train NN and plot training loss============
def train_and_plot_loss(y_ensemble, target_name):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    history = model.fit(X, y_ensemble, epochs=500, batch_size=4, verbose=0, callbacks=[es])
    loss = np.array(history.history['loss'])
    smooth_loss = np.convolve(loss, np.ones(10)/10, mode='valid')

    # ============ Title and Axis Label ============
    plt.figure(figsize=(6, 3.8))
    plt.plot(smooth_loss, linewidth=1.5)
    plt.xlabel("Epochs", fontsize=14, fontweight="bold")
    plt.ylabel("Mean Square Error", fontsize=14, fontweight="bold")
    plt.xticks(fontweight='bold',fontsize=12,fontname='Dejavu Sans')
    plt.yticks(fontweight='bold', fontsize=12, fontname='Dejavu Sans')
    plt.title(f"{target_name} - Training Loss", fontsize=16, fontweight="bold")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# ============ run for all datasets ============
for name, ensemble_values in datasets.items():
    train_and_plot_loss(ensemble_values, name)
