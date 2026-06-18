import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# ============ FUNCTION TO PLOT LEARNING CURVES ============
def plot_learning_curve(X, y, title):
    svr = SVR(kernel='rbf', C=10, epsilon=0.1)
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=0)
    xgb = XGBRegressor(random_state=0, verbosity=0)
    nn = MLPRegressor(random_state=0, max_iter=1000)

    ensemble = VotingRegressor([
        ('svr', svr),
        ('elastic net', elastic),
        ('xgb', xgb),
        ('nn', nn)
    ])

    train_sizes, train_scores, test_scores = learning_curve(
        ensemble,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
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

# ============ SURFACE ROUGHNESS DATA ======================
surface_roughness= {
    'Exp_value': [6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66, 4.13, 5.23, 6.26, 7.32,
                  4.90, 5.60, 6.44, 6.02, 4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69],
    'SVR': [6.17, 4.11, 6.65, 5.2, 5.91, 5.8, 5.95, 5.56, 4.23, 5.33, 6.324, 7.22,
            5, 5.601, 6.54, 5.92, 4.43, 5.913, 6.174, 7, 4.79, 6.53, 6.33, 4.79],
    'ElasticNet': [5.789, 4.371, 6.84, 5.268, 6.255, 5.565, 6.187, 5.318, 4.616, 5.445,
                   5.891, 7.085, 5.2, 6.011, 6.138, 5.684, 4.548, 5.667, 5.772,
                   6.908, 5.024, 6.664, 6.432, 4.792],
    'XGBoost': [6.071000099, 4.019000053, 6.748000145, 5.099999905, 6.008999825,
                5.900000095, 6.050000191, 5.659999847, 4.131999969, 5.230000019,
                6.257999897, 7.31099987, 4.900000095, 5.600999832, 6.436999798,
                6.020999908, 4.331999779, 5.940000057, 6.209000111, 7.098999977,
                4.691999912, 6.629000187, 6.427999973, 4.690000057],
    'NeuralNet': [6.152, 3.976, 6.874, 5.043, 6.167, 5.811, 6.086, 5.541, 4.154,
                  5.275, 6.221, 7.237, 4.965, 5.572, 6.23, 5.949, 4.349, 5.967,
                  6.055, 7.162, 4.79, 6.579, 6.475, 4.682]
}

df_sw = pd.DataFrame(surface_roughness)
X1 = df_sw[['SVR', 'ElasticNet', 'XGBoost', 'NeuralNet']].values
y1 = df_sw['Exp_value'].values

plot_learning_curve(X1, y1, "Learning Curve – Ensemble Surface Roughness")

# ============ MMR DATA ===================================

mmr = {
    'Exp_value': [100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.2, 95.14, 66.76, 82.97,
                  110.85, 115.82, 90.99, 102.68, 115.36, 102.48, 86.1, 107.2, 108.79,
                  122.52, 87.58, 114.14, 107, 78.57],
    'SVR': [102.291, 81.932, 116.089, 84.710, 93.370, 99.995, 110.849, 95.040,
            78.260, 86.750, 109.933, 113.627, 91.090, 102.580, 115.260, 102.380,
            86.200, 107.300, 107.953, 109.725, 87.680, 113.099, 106.900, 82.460],
    'ElasticNet': [97.725, 78.594, 125.440, 83.396, 102.282, 96.524, 114.915, 91.803,
                   72.871, 90.306, 101.787, 119.717, 96.028, 108.005, 106.508,
                   97.428, 85.504, 100.586, 100.883, 112.807, 89.118, 118.529,
                   109.193, 79.781],
    'XGBoost': [100.721, 74.823, 119.805, 79.165, 93.309, 99.863, 117.107, 95.154,
                66.975, 82.984, 110.833, 115.820, 90.993, 102.675, 115.337,
                102.477, 86.112, 107.198, 108.766, 122.374, 87.592, 114.171,
                106.994, 78.626],
    'NeuralNet': [102.098, 74.758, 120.265, 79.075, 93.261, 100.844, 117.311,
                  95.343, 66.798, 82.913, 110.229, 115.817, 91.173, 102.736,
                  114.397, 101.818, 86.046, 106.765, 108.063, 122.615,
                  87.665, 114.179, 106.915, 78.394]
}

df_mvr = pd.DataFrame(mmr)
X2 = df_mvr[['SVR', 'ElasticNet', 'XGBoost', 'NeuralNet']].values
y2 = df_mvr['Exp_value'].values

plot_learning_curve(X2, y2, "Learning Curve – Ensemble MMR")

# ========================================================
# ============ HAZ DATA ===================================
# ========================================================
haz = {
     'Exp_value': [90.05,71.88,102.71,85.07,89.82,88.42,89.89,88.12,74.46,85.36,
                  93.12,108.08,82.88,86.3,93.28,89.51,75.64,89.4,90.81,
                  106.4,78.76,94.76,94.59,81.59],
    'SVR': [90.775,74.55,98.807,85.17,89.72,87.731,89.79,85.703,76.635,
            85.46,93.02,99.937,82.98,86.4,94.626,89.41,76.468,89.3,
            90.91,98.974,78.86,94.66,94.49,81.689],
    'ElasticNet': [89.996,71.787,100.953,85.996,92.61,86.444,91.579,
                   84.791,76.622,89.8,91.131,105.788,84.965,87.775,
                   92.784,87.837,75.591,87.579,89.738,101.984,
                   81.162,97.15,96.413,80.425],
    'XGBoost': [90.06999969,71.98100281,102.6920013,85.0719986,89.81700134,
                88.4260025,89.88800049,88.125,74.48799896,85.37599945,
                93.09999847,107.9840012,82.88400269,86.31700134,93.25,
                89.50900269,75.66600037,89.41500092,90.79799652,
                106.3759995,78.78600311,94.76399994,94.56700134,
                81.58399963],
    'NeuralNet': [90.86,71.912,102.698,85.037,89.854,87.883,89.938,
                  87.961,74.531,85.398,93.322,108.01,82.881,86.415,
                  93.413,88.877,75.666,89.497,90.723,106.326,
                  78.792,94.779,94.545,81.576]
}

df_haz = pd.DataFrame(haz)
X3 = df_haz[['SVR', 'ElasticNet', 'XGBoost', 'NeuralNet']].values
y3 = df_haz['Exp_value'].values

plot_learning_curve(X3, y3, "Learning Curve – Ensemble HAZ")
