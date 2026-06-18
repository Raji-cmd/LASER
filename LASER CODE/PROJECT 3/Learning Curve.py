import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
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
# ============ SURFACE UNDULATION DATA =================
# ========================================================
surface_undulation = {
    'Exp_value': [6.07,4.01,6.75,5.10,6.01,5.90,6.05,5.66,4.13,5.23,6.26,7.32,
                  4.90,5.60,6.44,6.02,4.33,5.94,6.21,7.10,4.69,6.63,6.43,4.69],
    'Ridge': [5.819,4.128,6.965,5.265,6.26,5.534,6.191,5.286,4.491,5.559,
              5.921,7.327,5.196,5.897,6.17,5.654,4.422,5.637,5.801,7.034,
              4.902,6.671,6.554,4.785],
    'RandomForest': [6.075,4.11,6.798,4.97,5.993,5.973,6.066,5.766,4.228,5.136,
                     6.187,7.159,4.859,5.806,6.254,6.024,4.374,5.959,6.135,7.054,
                     4.706,6.74,6.335,4.554],
    'SVR': [6.17,4.11,6.65,5.2,5.91,5.8,5.95,5.56,4.23,5.33,6.324,7.22,
            5,5.601,6.54,5.92,4.43,5.913,6.174,7,4.79,6.53,6.33,4.79],
    'ElasticNet': [5.789,4.371,6.84,5.268,6.255,5.565,6.187,5.318,4.616,5.445,
                   5.891,7.085,5.2,6.011,6.138,5.684,4.548,5.667,5.772,6.908,
                   5.024,6.664,6.432,4.792]
}

df_su = pd.DataFrame(surface_undulation)
X_su = df_su[['Ridge', 'RandomForest', 'SVR', 'ElasticNet']].values
y_su = df_su['Exp_value'].values

plot_learning_curve(X_su, y_su, "Learning Curve – Ensemble Surface Undulation")

# ========================================================
# ============ MATERIAL REMOVAL RATE DATA ===============
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
    'SVR': [102.291,81.932,116.089,84.71,93.37,99.995,110.849,95.04,78.26,86.75,
            109.933,113.627,91.09,102.58,115.26,102.38,86.2,107.3,107.953,109.725,
            87.68,113.099,106.9,82.46],
    'ElasticNet': [97.725,78.594,125.44,83.396,102.282,96.524,114.915,91.803,
                   72.871,90.306,101.787,119.717,96.028,108.005,106.508,97.428,
                   85.504,100.586,100.883,112.807,89.118,118.529,109.193,79.781]
}

df_mvr = pd.DataFrame(mvr)
X_mvr = df_mvr[['Ridge', 'RandomForest', 'SVR', 'ElasticNet']].values
y_mvr = df_mvr['Exp_value'].values

plot_learning_curve(X_mvr, y_mvr, "Learning Curve – Ensemble Material Removal Rate")

# ========================================================
# ============ HAZ DATA ==================================
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
    'SVR': [90.775,74.55,98.807,85.17,89.72,87.731,89.79,85.703,76.635,85.46,
            93.02,99.937,82.98,86.4,94.626,89.41,76.468,89.3,90.91,98.974,
            78.86,94.66,94.49,81.689],
    'ElasticNet': [89.996,71.787,100.953,85.996,92.61,86.444,91.579,84.791,76.622,
                   89.8,91.131,105.788,84.965,87.775,92.784,87.837,75.591,87.579,
                   89.738,101.984,81.162,97.15,96.413,80.425]
}

df_haz = pd.DataFrame(haz)
X_haz = df_haz[['Ridge', 'RandomForest', 'SVR', 'ElasticNet']].values
y_haz = df_haz['Exp_value'].values

plot_learning_curve(X_haz, y_haz, "Learning Curve – Ensemble HAZ")