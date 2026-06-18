from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
external_surface_roughness = pd.DataFrame({
    'Actual': [
        6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
        4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
        4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69
    ],
    'Predicted': [
        6.054, 4.093, 6.795, 5.116, 6.067, 5.807, 6.068, 5.566, 4.259, 5.295, 6.179, 7.227, 4.984, 5.707
        , 6.323, 5.916, 4.394, 5.886, 6.072, 7.057, 4.788, 6.629, 6.412, 4.701
    ]
})

material_removal_quantity = pd.DataFrame({
    'Actual': [
        100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
        66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
        86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57
    ],
    'Predicted': [
         101.16, 77.37, 120.03, 80.90, 95.76, 99.76, 114.76, 95.06, 70.94, 85.12, 108.39, 116.43,
         91.69, 103.77, 112.34, 101.20, 85.50, 105.68, 106.52, 117.09, 87.33, 115.22, 107.58, 79.59,
    ]
})

HAZ = pd.DataFrame({
    'Actual': [
        90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
        74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
        75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59
    ],
    'Predicted': [
        90.337, 72.734, 101.499, 84.786, 90.456, 87.866, 90.365, 87.044, 75.572, 86.161, 92.538, 105.676, 83.136,
        86.988, 93.153, 88.969, 76.033, 88.999, 90.520, 103.685, 79.428, 96.158, 94.727, 81.066,
    ]
})

# ============ Metric function ============
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ============ Compute Metrics ============
surface_metrics = get_metrics(external_surface_roughness['Actual'], external_surface_roughness['Predicted'])
mrq_metrics = get_metrics(material_removal_quantity['Actual'], material_removal_quantity['Predicted'])
haz_metrics = get_metrics(HAZ['Actual'], HAZ['Predicted'])

# ============ Final Output ============
print("\n=========== Ensemble Model Performance ===========")
print(f"{'Target':<35}{'RMSE':>12}{'MAE':>12}{'R²':>12}")
print(f"{'External Surface Roughness':<35}{surface_metrics[0]:>12.4f}{surface_metrics[1]:>12.4f}{surface_metrics[2]:>12.4f}")
print(f"{'Material Removal Quantity':<35}{mrq_metrics[0]:>12.4f}{mrq_metrics[1]:>12.4f}{mrq_metrics[2]:>12.4f}")
print(f"{'Heat Affected Zone ':<35}{haz_metrics[0]:>12.4f}{haz_metrics[1]:>12.4f}{haz_metrics[2]:>12.4f}")
