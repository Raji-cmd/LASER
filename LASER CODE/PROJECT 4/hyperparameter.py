import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor

# -------------------------
# Data
# -------------------------
data = {
    "PLZ": [8.5, 6.5, 10.5, 6.5, 6.5, 8.5, 10.5, 8.5, 6.5, 10.5, 8.5, 10.5,
            10.5, 6.5, 8.5, 7.5, 10.5, 8.5, 9.5, 6.5, 6.5, 6.5, 10.5, 10.5],
    "SSP": [280, 360, 200, 360, 200, 280, 200, 320, 360, 360, 280, 200,
            360, 200, 240, 280, 360, 280, 280, 200, 360, 200, 200, 360],
    "PC": [24, 12, 36, 36, 12, 18, 12, 24, 12, 36, 30, 36,
           36, 12, 24, 24, 12, 24, 24, 36, 36, 36, 12, 12],
    "FLZ": [48, 18, 18, 58, 58, 38, 18, 38, 58, 58, 38, 58,
            18, 18, 38, 38, 18, 28, 38, 58, 18, 18, 58, 58],

    "ESR": [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66, 4.13, 5.23, 6.26, 7.32,
           4.90, 5.60, 6.44, 6.02, 4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69],

    "MRQ": [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14, 66.76,
            82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48, 86.10,
            107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57],

    "HAZ": [90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12, 74.46,
            85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51, 75.64,
            89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59]
}
df = pd.DataFrame(data)

input_features = ["PLZ", "SSP", "PC", "FLZ"]
targets = ["ESR", "MRQ", "HAZ"]

# -------------------------
# Neural Network Builder
# -------------------------
def build_nn():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(4,)),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

# -------------------------
# Model Configurations
# -------------------------
def get_model_configs():
    return {
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

# -------------------------
# Training
# -------------------------
results = {}

print(f"Training on ALL {len(df)} samples...")
print("=" * 50)

for target in targets:

    X = df[input_features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_configs = get_model_configs()
    row = {}

    for name, model in model_configs.items():
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = round(r2_score(y, y_pred), 3)

        try:
            hyperparams = model.get_params()
        except:
            hyperparams = {"model": "Keras Sequential"}

        row[name] = {"hyperparameters": hyperparams, "R²": r2}
        print(f"{target} - {name}: R² = {r2}")

    results[target] = row
    print("-" * 40)

# -------------------------
# Save Results
# -------------------------
flattened_rows = []

for target, model_dict in results.items():
    row = {"Output": target}

    for model_name, metrics in model_dict.items():
        hyperparams = metrics["hyperparameters"]
        row[f"{model_name} Hyperparameter"] = str(hyperparams)
        row[f"{model_name} R²"] = metrics["R²"]

    flattened_rows.append(row)

final_excel_df = pd.DataFrame(flattened_rows)
final_excel_df.to_excel("Hyperparameter_All_Data_With_XGB_NN.xlsx", index=False)

print("\n✅ Results saved to 'Hyperparameter_All_Data_With_XGB_NN.xlsx'")
print("\nFinal Summary:")
print(final_excel_df.to_string(index=False))