import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
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
exp_nos = list(range(1, len(df) + 1))

# ============ Input & Target ============
X = df[['PLZ', 'SSP','PC', 'FLZ']]
targets = {
    'Surface Undulation': df['Surface Undulation'],
    'Material Removal Rate': df['Material Removal Rate'],
    'Heat Affected Zone': df['Heat Affected Zone']
}

# ============ Models ============
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=120, random_state=42),
    'SVR': SVR(C=10, epsilon=0.1, kernel='rbf'),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42)
}

colors = ['red', 'blue', 'green', 'orange']

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