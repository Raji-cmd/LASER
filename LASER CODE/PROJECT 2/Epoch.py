import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

sr_actual = np.array([
    6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
    4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
    4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69
])

mmr_actual = np.array([
    100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
    66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
    86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57
])

haz_actual = np.array([
    90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
    74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
    75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59
])

# ============ Predicted values ============
sr_pred = np.array([6.039,4.108,6.794,5.14,6.103,5.774,6.08,5.524,4.274,
                    5.312,6.165,7.218,5.01,5.695,6.315,5.9,4.406,5.876,6.048,7.055,
                    4.819,6.608,6.429,4.728,
])

mmr_pred = np.array([ 100.621, 76.813, 120.794, 81.028, 95.547, 99.342, 115.681,94.356, 70.089, 85.358,
                      108.255, 116.464, 92.321, 104.011, 112.836, 101.002, 85.953, 105.430,
                      106.462, 118.157, 88.009, 115.102, 107.506, 79.420
])

haz_pred = np.array([90.394, 72.297, 101.676, 85.307, 90.512, 87.663, 90.311, 86.879, 75.356, 86.501,
                     92.662, 106.236, 83.418, 86.723, 93.389, 88.886,75.768,88.963,90.527,
                     104.153,79.393,95.349,95.011,81.308,

])

# ============ Ensemble ============
datasets = {
    "Surface Roughness": (sr_actual + sr_pred) / 2,
    "MMR": (mmr_actual + mmr_pred) / 2,
    "HAZ":(haz_actual + haz_pred) /2,
}
# ============ sequence index ============
num_samples = len(sr_actual)
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
