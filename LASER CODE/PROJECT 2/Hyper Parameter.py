import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

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

    "SR": [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66, 4.13, 5.23, 6.26, 7.32,
           4.90, 5.60, 6.44, 6.02, 4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69],

    "MMR": [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14, 66.76,
            82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48, 86.10,
            107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57],

    "HAZ": [90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12, 74.46,
            85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51, 75.64,
            89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59]
}

df = pd.DataFrame(data)

# ============ Inputs and Outputs ============
input_features = ["PLZ", "SSP", "PC", "FLZ"]
targets = ["SR", "MMR", "HAZ"]

# ============ Model Configurations ============
model_configs = {
    "SR": {
        "SVR": SVR(C=10, epsilon=0.1, kernel='rbf'),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42),
        "XGB": XGBRegressor(n_estimators=200, learning_rate=0.05,max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, random_state=42),
        "NN": MLPRegressor(hidden_layer_sizes=(16, 8, 4),
                           activation='relu', max_iter=2000,
                           random_state=42)
    },

    "MMR": {
        "SVR": SVR(C=10, epsilon=0.1, kernel='rbf'),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42),
        "XGB": XGBRegressor(n_estimators=200, learning_rate=0.05,
                            max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, random_state=42),
        "NN": MLPRegressor(hidden_layer_sizes=(16, 10, 6),
                           activation='relu', max_iter=2000,
                           random_state=42)
    },

    "HAZ": {
        "SVR": SVR(C=10, epsilon=0.1, kernel='rbf'),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42),
        "XGB": XGBRegressor(n_estimators=200, learning_rate=0.05,
                            max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, random_state=42),
        "NN": MLPRegressor(hidden_layer_sizes=(12, 6, 3),
                           activation='relu', max_iter=2000,
                           random_state=42)
    }
}

# ============ Model Training ============
results = {}
print("Training individual models (SVR,Elastic net, XGB, NN)")
print("=" * 60)

for target in targets:
    X = df[input_features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    row = {}
    for name, model in model_configs[target].items():
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = round(r2_score(y, y_pred), 3)

        row[name] = {"Hyperparameters": model.get_params(), "R²": r2}
        print(f"{target} → {name}: R² = {r2}")

    results[target] = row
    print("-" * 60)

# ============ Save to Excel ============
flatten = []
for target, model_dict in results.items():
    row = {"Output": target}
    for model_name, metrics in model_dict.items():
        row[f"{model_name} Hyperparameters"] = str(metrics["Hyperparameters"])
        row[f"{model_name} R²"] = metrics["R²"]
    flatten.append(row)

df_out = pd.DataFrame(flatten)
df_out.to_excel("Individual_Model_Results.xlsx", index=False)

print("\nSaved to Individual_Model_Results.xlsx")
print(df_out)
