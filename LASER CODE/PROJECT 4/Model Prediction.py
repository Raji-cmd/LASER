import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
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
     "External Surface Roughness": [6.054, 4.093, 6.795, 5.116, 6.067, 5.807, 6.068, 5.566, 4.259, 5.295, 6.179, 7.227, 4.984, 5.707
        , 6.323, 5.916, 4.394, 5.886, 6.072, 7.057, 4.788, 6.629, 6.412, 4.701],

    "Material Removal Quantity": [101.16, 77.37, 120.03, 80.90, 95.76, 99.76, 114.76, 95.06, 70.94, 85.12, 108.39, 116.43,
         91.69, 103.77, 112.34, 101.20, 85.50, 105.68, 106.52, 117.09, 87.33, 115.22, 107.58, 79.59],

    "Heat Affected Zone": [90.337, 72.734, 101.499, 84.786, 90.456, 87.866, 90.365, 87.044, 75.572, 86.161, 92.538, 105.676, 83.136,
        86.988, 93.153, 88.969, 76.033, 88.999, 90.520, 103.685, 79.428, 96.158, 94.727, 81.066]
}

df = pd.DataFrame(data)
exp_nos = list(range(1, len(df) + 1))

# ============ Input & Target ============
X = df[['PLZ', 'SSP','PC', 'FLZ']]
targets = {
    'External Surface Roughness': df['External Surface Roughness'],
    'Material Removal Quantity': df['Material Removal Quantity'],
    'Heat Affected Zone': df['Heat Affected Zone']
}

# ============ Neural Network Function ============
def build_nn():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(4,)),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='mse'
    )
    return model
# ============ Models ============
models = {
    "Ridge": Ridge(alpha=1),
    "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42),
    "SVR": SVR(C=10, epsilon=0.1, kernel='rbf'),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    ),
    "NeuralNet": KerasRegressor(
        model=build_nn,
        epochs=200,
        batch_size=8,
        verbose=0
    )
}

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

# ============ Function To Generate Full Predictions ============
def get_predictions(X, y):
    predictions = {}

    for name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale for models sensitive to feature scaling
        if name in ['SVR', 'ElasticNet']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Combine predictions: keep training data actual values
        full_pred = np.zeros(len(X))
        full_pred[X_test.index] = y_pred
        full_pred[X_train.index] = y.iloc[X_train.index].values
        predictions[name] = full_pred

    return predictions

# ============ Plot Figures ============
for target_name, y in targets.items():
    predictions = get_predictions(X, y)
    plt.figure(figsize=(12, 6))

    # Plot experimental values
    plt.plot(exp_nos, y, 'ko-', linewidth=3, markersize=7, label='Experimental', markerfacecolor='black')

    # Plot model predictions
    for i, (name, pred) in enumerate(predictions.items()):
        plt.plot(exp_nos, pred, 's-', color=colors[i], markersize=5,
                 label=name, alpha=0.85, linewidth=2)

    plt.title(f"{target_name} - Model Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Experiment Number", fontsize=14, fontweight='bold')
    plt.ylabel(target_name, fontsize=14, fontweight='bold')
    plt.xticks(fontweight='bold', fontsize=12)
    plt.yticks(fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=10)
    plt.show()