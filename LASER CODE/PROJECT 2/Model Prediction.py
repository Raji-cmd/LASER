import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============ Dataset ============
data = {
    "PLZ": [8.5, 6.5, 10.5, 6.5, 6.5, 8.5, 10.5, 8.5, 6.5, 10.5, 8.5, 10.5,
            10.5, 6.5, 8.5, 7.5, 10.5, 8.5, 9.5, 6.5, 6.5, 6.5, 10.5, 10.5],
    "SSP": [280, 360, 200, 360, 200, 280, 200, 320, 360, 360, 280, 200,
            360, 200, 240, 280, 360, 280, 280, 200, 360, 200, 200, 360],
    "PC": [24, 12, 36, 36, 12, 18, 12, 24, 12, 36, 30, 36,
           36, 12, 24, 24, 12, 24, 24, 36, 36, 36, 12, 12],
    "FLZ": [48, 18, 18, 58, 58, 38, 18, 38, 58, 58, 38, 58,
            18, 18, 38, 38, 18, 28, 38, 58, 18, 18, 58, 58],

    "Surface Roughness": [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66, 4.13, 5.23,
                          6.26, 7.32, 4.90, 5.60, 6.44, 6.02, 4.33, 5.94, 6.21,
                          7.10, 4.69, 6.63, 6.43, 4.69],

    "Material Melting Rate": [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20,
                              95.14, 66.76, 82.97, 110.85, 115.82, 90.99, 102.68,
                              115.36, 102.48, 86.10, 107.20, 108.79, 122.52,
                              87.58, 114.14, 107.00, 78.57],

    "Heat Affected Zone": [90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
                           74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28,
                           89.51, 75.64, 89.40, 90.81, 106.40, 78.76, 94.76,
                           94.59, 81.59]
}

df = pd.DataFrame(data)
exp_nos = np.arange(1, len(df) + 1)

# ============ Inputs & Targets ============
X = df[['PLZ', 'SSP', 'PC', 'FLZ']]

targets = {
    'Surface Roughness': df['Surface Roughness'],
    'Material Melting Rate': df['Material Melting Rate'],
    'Heat Affected Zone': df['Heat Affected Zone']
}

# ============ Models ============
models = {
    'SVR': SVR(kernel='rbf', C=10, epsilon=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=0),
    'XGBoost': XGBRegressor(n_estimators=120, random_state=42, verbosity=0)
}

# ============ Neural Networks ============
nn_models = {
    'Surface Roughness': MLPRegressor(hidden_layer_sizes=(6,4,3), max_iter=2000, random_state=42),
    'Material Melting Rate': MLPRegressor(hidden_layer_sizes=(8,5,4), max_iter=2000, random_state=42),
    'Heat Affected Zone': MLPRegressor(hidden_layer_sizes=(6,4,2), max_iter=2000, random_state=42)
}

colors = ['red', 'blue', 'green', 'orange']

# ============ Prediction Function ============
def get_predictions(X, y, target_name):
    predictions = {}

    full_models = models.copy()
    full_models['Neural Network'] = nn_models[target_name]

    for name, model in full_models.items():

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        if name == 'Neural Network':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        full_pred = np.zeros(len(X))
        full_pred[X_train.index] = y.iloc[X_train.index].values
        full_pred[X_test.index] = y_pred

        predictions[name] = full_pred

    return predictions

# ============ Plotting ============
for target_name, y in targets.items():
    predictions = get_predictions(X, y, target_name)

    plt.figure(figsize=(12,6))

    plt.plot(exp_nos, y, 'ko-', linewidth=3, markersize=7,
             label='Experimental', markerfacecolor='black')

    for i, (name, pred) in enumerate(predictions.items()):
        plt.plot(exp_nos, pred, 's-', color=colors[i],
                 markersize=5, linewidth=2, label=name, alpha=0.85)

    plt.title(f"{target_name} – Model Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Experiment Number", fontsize=14, fontweight='bold')
    plt.ylabel(target_name, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
