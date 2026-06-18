from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# ============ Actual values ============
sr_actual = np.array([
    6.07, 4.01, 6.75, 5.10, 6.01, 5.90, 6.05, 5.66,
    4.13, 5.23, 6.26, 7.32, 4.90, 5.60, 6.44, 6.02,
    4.33, 5.94, 6.21, 7.10, 4.69, 6.63, 6.43, 4.69
])

mmr_actual = np.array([
    100.71, 74.77, 119.88, 79.11, 93.27, 99.84, 117.20, 95.14,
    66.76, 82.97, 110.85, 115.82, 90.99, 102.68, 115.36, 102.48,
    86.10, 107.20, 108.79, 122.52, 87.58, 114.14, 107.00, 78.57
])

haz_actual = np.array([
    90.05, 71.88, 102.71, 85.07, 89.82, 88.42, 89.89, 88.12,
    74.46, 85.36, 93.12, 108.08, 82.88, 86.30, 93.28, 89.51,
    75.64, 89.40, 90.81, 106.40, 78.76, 94.76, 94.59, 81.59
])

# ============ Predicted values ============
sr_pred = np.array([6.039,4.108,6.794,5.14,6.103,5.774,6.08,5.524,4.274,
                    5.312,6.165,7.218,5.01,5.695,6.315,5.9,4.406,5.876,6.048,7.055,
                    4.819,6.608,6.429,4.728,
])

mmr_pred = np.array([ 100.621, 76.813, 120.794, 81.028, 95.547, 99.342, 115.681,94.356, 70.089, 85.358,
                      108.255, 116.464, 92.321, 104.011, 112.836, 101.002, 85.953, 105.430,
                      106.462, 118.157, 88.009, 115.102, 107.506, 79.420
])

haz_pred = np.array([90.394, 72.297, 101.676, 85.307, 90.512, 87.663, 90.311, 86.879, 75.356, 86.501,
                     92.662, 106.236, 83.418, 86.723, 93.389, 88.886,75.768,88.963,90.527,
                     104.153,79.393,95.349,95.011,81.308,

])

# ============ Metric function ============
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ============ Compute Metrics ============
sr_metrics = get_metrics(sr_actual, sr_pred)
mmr_metrics = get_metrics(mmr_actual, mmr_pred)
haz_metrics = get_metrics(haz_actual, haz_pred)

# ============ Final Output ============
print("\n=========== Ensemble Model Performance ===========")
print(f"{'Target':<30}{'RMSE':>12}{'MAE':>12}{'R²':>12}")
print(f"{'Surface Roughness':<30}{sr_metrics[0]:>12.4f}{sr_metrics[1]:>12.4f}{sr_metrics[2]:>12.4f}")
print(f"{'Material Melting Rate':<30}{mmr_metrics[0]:>12.4f}{mmr_metrics[1]:>12.4f}{mmr_metrics[2]:>12.4f}")
print(f"{'Heat Affected Zone':<30}{haz_metrics[0]:>12.4f}{haz_metrics[1]:>12.4f}{haz_metrics[2]:>12.4f}")
