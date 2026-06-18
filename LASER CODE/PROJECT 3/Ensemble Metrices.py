from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

surface_undulation = pd.DataFrame({
    'Actual': [
        6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
        4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
        4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69
    ],
    'Predicted': [
        5.979, 4.191, 6.791, 5.173, 6.087, 5.733, 6.086, 5.498,
        4.384, 5.35, 6.1, 7.18, 5.054, 5.82, 6.292, 5.835,
        4.45, 5.809, 5.988, 6.991, 4.856, 6.644, 6.395, 4.731
    ]
})

material_removal_rate = pd.DataFrame({
    'Actual': [
        100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
        66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
        86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57
    ],
    'Predicted': [
        100.401, 79.131, 120.923, 82.47, 98.176, 98.832, 113.374, 94.31,
        73.706, 87.317, 106.071, 117.261, 92.83, 105.097, 110.057, 100.027,
        85.243, 104.087, 104.521, 113.161, 87.554, 116.303, 108.177, 80.311
    ]
})

HAZ = pd.DataFrame({
    'Actual': [
        90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
        74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
        75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59
    ],
    'Predicted': [
        90.228, 73.02, 100.662, 84.913, 91.161, 87.43, 90.805, 86.026,
        76.361, 87.244, 91.945, 104.26, 83.62, 87.424, 93.065, 88.651,
        76.144, 88.5, 90.263, 101.782, 80.061, 96.977, 95.142, 80.711
    ]
})

# ============ Metric function ============
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ============ Compute Metrics ============
surface_metrics = get_metrics(surface_undulation['Actual'], surface_undulation['Predicted'])
mrr_metrics = get_metrics(material_removal_rate['Actual'], material_removal_rate['Predicted'])
haz_metrics = get_metrics(HAZ['Actual'], HAZ['Predicted'])

# ============ Final Output ============
print("\n=========== Ensemble Model Performance ===========")
print(f"{'Target':<35}{'RMSE':>12}{'MAE':>12}{'R²':>12}")
print(f"{'Surface Undulation':<35}{surface_metrics[0]:>12.4f}{surface_metrics[1]:>12.4f}{surface_metrics[2]:>12.4f}")
print(f"{'Material Removal Rate (MRR)':<35}{mrr_metrics[0]:>12.4f}{mrr_metrics[1]:>12.4f}{mrr_metrics[2]:>12.4f}")
print(f"{'Heat Affected Zone (HAZ)':<35}{haz_metrics[0]:>12.4f}{haz_metrics[1]:>12.4f}{haz_metrics[2]:>12.4f}")
