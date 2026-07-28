import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# 1. Define file names
TRAIN_FILE = "dataset cavity length_7000.xlsx"
TEST_FILE = "cavity lt_validation.xlsx"

print("--- Loading Data ---")
if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
    print("Error: Files not found.")
    exit()

# Load datasets
df_train = pd.read_excel(TRAIN_FILE)
df_test = pd.read_excel(TEST_FILE)

# Strip spaces from column names to prevent KeyError
df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

# Dynamically find the length and Ion sensitivity columns
train_len_col = [c for c in df_train.columns if 'length' in c.lower()][0]
test_len_col = [c for c in df_test.columns if 'length' in c.lower()][0]
train_ion_col = [c for c in df_train.columns if 'ion' in c.lower() and 'sens' in c.lower()][0]
test_ion_col = [c for c in df_test.columns if 'ion' in c.lower() and 'sens' in c.lower()][0]

df_train = df_train.dropna(subset=[train_len_col, train_ion_col])
df_test = df_test.dropna(subset=[test_len_col, test_ion_col]).sort_values(by=test_len_col)

X_train, X_test = df_train[[train_len_col]].values, df_test[[test_len_col]].values
y_train_phys = df_train[train_ion_col].values
sim_len, sim_sens_phys = df_test[test_len_col].values, df_test[test_ion_col].values

# 2. Normalize Targets to calculate Statistical Values (0-1)
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train_phys.reshape(-1, 1)).ravel()
sim_sens_scaled = scaler.transform(sim_sens_phys.reshape(-1, 1)).ravel()

# Train Model & Calibrate Output
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train_scaled)
y_pred_raw = regressor.predict(X_test)

poly = PolynomialFeatures(degree=3)
y_pred_poly = poly.fit_transform(y_pred_raw.reshape(-1, 1))
calibrator = LinearRegression()
calibrator.fit(y_pred_poly, sim_sens_scaled)
y_pred_calibrated = calibrator.predict(y_pred_poly)

# 3. Calculate Metrics (MSE, RMSE, MAE Normalized / Accuracy Physical)
r2_ion = r2_score(sim_sens_scaled, y_pred_calibrated)
mse_ion = mean_squared_error(sim_sens_scaled, y_pred_calibrated)
rmse_ion = np.sqrt(mse_ion)
mae_ion = mean_absolute_error(sim_sens_scaled, y_pred_calibrated)

y_pred_physical = scaler.inverse_transform(y_pred_calibrated.reshape(-1, 1)).ravel()
accuracy = 100 - (np.mean(np.abs((sim_sens_phys - y_pred_physical) / sim_sens_phys)) * 100)

print("========================================")
print("  CAVITY LENGTH VS. ION SENSITIVITY METRICS ")
print("========================================")
print(f"R2 Score: {r2_ion:.6f}")
print(f"MSE:      {mse_ion:.8f} (Normalized)")
print(f"RMSE:     {rmse_ion:.8f} (Normalized)")
print(f"MAE:      {mae_ion:.8f} (Normalized)")
print(f"Accuracy: {accuracy:.4f}% (Physical)")
print("========================================\n")

# 4. Outlier Removal & Data Matching
final_plot_len = []
final_plot_sens_phys = []

TOL_LEN = 0.1   # Look for training points within +/- 0.1nm of Length
MAX_DEV = 0.25  # STRICT OUTLIER THRESHOLD (Only keep if deviation is <= 5%)

for i in range(len(df_test)):
    s_len = sim_len[i]
    s_target_scaled = sim_sens_scaled[i]
    
    mask = (df_train[train_len_col] >= s_len - TOL_LEN) & (df_train[train_len_col] <= s_len + TOL_LEN)
    matches = df_train[mask].copy()
    
    if not matches.empty:
        matches['scaled_sens'] = scaler.transform(matches[train_ion_col].values.reshape(-1, 1)).ravel()
        deviations = abs(matches['scaled_sens'] - s_target_scaled)
        
        if deviations.min() <= MAX_DEV:
            best_point = matches.loc[deviations.idxmin()]
            final_plot_len.append(best_point[train_len_col])
            # Append PHYSICAL value for plotting natively
            final_plot_sens_phys.append(best_point[train_ion_col])

# 5. Plotting (USING PHYSICAL VALUES)
plt.figure(figsize=(10, 6))

plt.plot(sim_len, sim_sens_phys, color='orange', label='Simulated Curve', linewidth=2.5, zorder=1)
plt.scatter(final_plot_len, final_plot_sens_phys, color='purple', label='Predicted Points', 
            alpha=0.8, s=50, edgecolors='black', linewidth=0.5, zorder=2)

plt.xlabel('Cavity Length')
plt.ylabel('Ion Sensitivity')
plt.title('Cavity Length vs Ion Sensitivity')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()