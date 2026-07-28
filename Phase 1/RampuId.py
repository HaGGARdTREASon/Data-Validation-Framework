import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load Data
train_file = "dataset rampup_8000_nw.xlsx"
test_file = "ramp up curves-testing.xlsx"

df_train = pd.read_excel(train_file).dropna(subset=['Width', 'Id sensitivity'])
df_test = pd.read_excel(test_file).dropna(subset=['Width', 'Id sensitivity']).sort_values(by='Width')

X_train, X_test = df_train[['Width']].values, df_test[['Width']].values
y_train_phys = df_train['Id sensitivity'].values.reshape(-1, 1)
y_test_phys = df_test['Id sensitivity'].values.reshape(-1, 1)

# 2. Normalize & Train
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train_phys).ravel()
y_test_scaled = scaler.transform(y_test_phys).ravel()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_scaled)

# 3. Calculate Metrics (calculated on scaled data for consistency, accuracy on physical)
y_pred_scaled = rf.predict(X_test)
r2_id = r2_score(y_test_scaled, y_pred_scaled)
mse_id = mean_squared_error(y_test_scaled, y_pred_scaled)
rmse_id = np.sqrt(mse_id)
mae_id = mean_absolute_error(y_test_scaled, y_pred_scaled)

# Inverse transform to get physical predictions for calculating real Accuracy
y_pred_phys = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
acc_id = 100 - (np.mean(np.abs((y_test_phys.ravel() - y_pred_phys) / y_test_phys.ravel())) * 100)

# Print Block
print("========================================")
print("    WIDTH VS. ID SENSITIVITY METRICS    ")
print("========================================")
print(f"R2 Score: {r2_id:.6f}")
print(f"MSE:      {mse_id:.8f} (Normalized)")
print(f"RMSE:     {rmse_id:.8f} (Normalized)")
print(f"MAE:      {mae_id:.8f} (Normalized)")
print(f"Accuracy: {acc_id:.4f}% (Physical)")
print("========================================\n")

# 4. Plot Training Predictions vs Test Curve (USING PHYSICAL VALUES)
plt.figure(figsize=(5, 4))

# Plot simulated curve (test data) using ORIGINAL PHYSICAL VALUES
plt.plot(X_test, y_test_phys, color='blue', label='Simulated Curve', linewidth=2, zorder=1)

# Predict on training data and INVERSE TRANSFORM back to physical values
y_train_pred_scaled = rf.predict(X_train)
y_train_pred_phys = scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

# Filter relevant training points: within test range
train_mask = (X_train.ravel() >= X_test.min()) & (X_train.ravel() <= X_test.max())
X_train_relevant = X_train[train_mask].ravel()
y_train_pred_relevant = y_train_pred_phys[train_mask] # Use physical predictions here

# Apply tolerance filter (minimum spacing between points)
tolerance = 0.08   # adjust as needed
X_filtered = []
y_filtered = []

last_x = -np.inf
for x, y in zip(X_train_relevant, y_train_pred_relevant):
    if x - last_x >= tolerance:
        X_filtered.append(x)
        y_filtered.append(y)
        last_x = x

# Plot filtered predicted points
plt.scatter(X_filtered, y_filtered, 
            color='red', alpha=0.6, s=25, label='Predicted Points', zorder=2)

plt.xlabel('Width(nm)')
plt.ylabel('Id Sensitivity')
plt.title('Rampup: Width vs Id Sensitivity')
plt.legend()
plt.grid(True)
plt.show()