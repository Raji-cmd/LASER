import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# ============ FUNCTION TO PLOT LEARNING CURVES ============
def plot_learning_curve(X, y, title):
    ridge = Ridge()
    rf = RandomForestRegressor(random_state=0)
    xgb = XGBRegressor(random_state=0, verbosity=0)
    nn = MLPRegressor(random_state=0, max_iter=1000)

    ensemble = VotingRegressor([
        ('ridge', ridge),
        ('rf', rf),
        ('xgb', xgb),
        ('nn', nn)
    ])

    train_sizes, train_scores, test_scores = learning_curve(
        ensemble, X, y,
        cv=5,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5)
    )

    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Error')
    plt.plot(train_sizes, test_mean, 'o-', label='Validation Error')

    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.15)
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.15)

    plt.xlabel("Training Samples", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Squared Error", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# ========================================================
# ============ SURFACE WAVINESS DATA ======================
# ========================================================
surface_waviness = {
    'Exp_value': [6.07,4.01,6.75,5.10,6.01,5.90,6.05,5.66,4.13,5.23,6.26,7.32,
                  4.90,5.60,6.44,6.02,4.33,5.94,6.21,7.10,4.69,6.63,6.43,4.69],
    'Ridge': [5.819,4.128,6.965,5.265,6.26,5.534,6.191,5.286,4.491,5.559,5.921,
              7.327,5.196,5.897,6.17,5.654,4.422,5.637,5.801,7.034,4.902,6.671,6.554,4.785],
    'RandomForest': [6.075,4.11,6.798,4.97,5.993,5.973,6.066,5.766,4.228,5.136,
                     6.187,7.159,4.859,5.806,6.254,6.024,4.374,5.959,6.135,7.054,4.706,6.74,6.335,4.554],
    'XGBoost': [6.071,4.019,6.748,5.10,6.009,5.90,6.05,5.66,4.132,5.23,6.258,7.311,
                4.90,5.601,6.437,6.021,4.332,5.94,6.209,7.099,4.692,6.629,6.428,4.69],
    'NeuralNet': [6.152,3.976,6.874,5.043,6.167,5.811,6.086,5.541,4.154,5.275,
                  6.221,7.237,4.965,5.572,6.23,5.949,4.349,5.967,6.055,7.162,4.79,6.579,6.475,4.682]
}

df_sw = pd.DataFrame(surface_waviness)
X1 = df_sw[['Ridge', 'RandomForest', 'XGBoost', 'NeuralNet']].values
y1 = df_sw['Exp_value'].values

plot_learning_curve(X1, y1, "Learning Curve – Ensemble Surface Waviness")

# ========================================================
# ============ MVR DATA ===================================
# ========================================================
mvr = {
    'Exp_value': [100.71,74.77,119.88,79.11,93.27,99.84,117.2,95.14,66.76,82.97,
                  110.85,115.82,90.99,102.68,115.36,102.48,86.1,107.2,108.79,
                  122.52,87.58,114.14,107,78.57],
    'Ridge': [97.684,78.313,125.882,83.154,102.192,96.474,115.157,91.715,
              72.429,90.235,101.837,119.998,96.119,108.076,106.596,97.385,
              85.395,100.626,100.926,112.916,89.038,118.8,109.273,79.511],
    'RandomForest': [102.973,76.905,118.33,78.26,96.607,101.619,113.485,97.999,
                     70.008,82.662,109.119,117.033,89.104,102.84,110.15,101.906,
                     83.69,106.509,106.926,117.854,84.638,115.979,107.834,78.846],
    'XGBoost': [100.721,74.823,119.805,79.165,93.309,99.863,117.107,95.154,
                66.975,82.984,110.833,115.82,90.993,102.675,115.337,102.477,
                86.112,107.198,108.766,122.374,87.592,114.171,106.994,78.626],
    'NeuralNet': [102.098,74.758,120.265,79.075,93.261,100.844,117.311,95.343,
                  66.798,82.913,110.229,115.817,91.173,102.736,114.397,101.818,
                  86.046,106.765,108.063,122.615,87.665,114.179,106.915,78.394]
}

df_mvr = pd.DataFrame(mvr)
X2 = df_mvr[['Ridge', 'RandomForest', 'XGBoost', 'NeuralNet']].values
y2 = df_mvr['Exp_value'].values

plot_learning_curve(X2, y2, "Learning Curve – Ensemble MVR")

# ========================================================
# ============ HAZ DATA ===================================
# ========================================================
haz = {
    'Exp_value': [90.05,71.88,102.71,85.07,89.82,88.42,89.89,88.12,74.46,85.36,
                  93.12,108.08,82.88,86.3,93.28,89.51,75.64,89.4,90.81,106.4,
                  78.76,94.76,94.59,81.59],
    'Ridge': [90.035,71.42,101.166,85.974,92.641,86.396,91.601,84.73,76.409,
              89.924,91.179,106.155,84.934,87.652,92.845,87.8,75.369,87.54,
              89.775,102.206,80.985,97.217,96.59,80.358],
    'RandomForest': [89.965,73.623,102.238,82.668,90.264,88.872,90.618,88.673,
                     75.69,84.708,92.092,106.476,81.985,88.094,91.66,89.23,
                     76.882,89.222,90.41,104.653,79.627,99.405,93.529,80.09],
    'XGBoost': [90.07,71.981,102.692,85.072,89.817,88.426,89.888,88.125,
                74.488,85.376,93.10,107.984,82.884,86.317,93.25,89.509,
                75.666,89.415,90.798,106.376,78.786,94.764,94.567,81.584],
    'NeuralNet': [90.86,71.912,102.698,85.037,89.854,87.883,89.938,87.961,
                  74.531,85.398,93.322,108.01,82.881,86.415,93.413,88.877,
                  75.666,89.497,90.723,106.326,78.792,94.779,94.545,81.576]
}

df_haz = pd.DataFrame(haz)
X3 = df_haz[['Ridge', 'RandomForest', 'XGBoost', 'NeuralNet']].values
y3 = df_haz['Exp_value'].values

plot_learning_curve(X3, y3, "Learning Curve – Ensemble HAZ")
