import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load Data
train_file = "dataset concave_8000_nw.xlsx"
test_file = "Concave curve-testing.xlsx"

df_train = pd.read_excel(train_file).dropna(subset=['Width', 'id sensitivity'])
df_test = pd.read_excel(test_file).dropna(subset=['Width', 'id sensitivity']).sort_values(by='Width')

X_train, X_test = df_train[['Width']].values, df_test[['Width']].values
y_train_phys = df_train['id sensitivity'].values.reshape(-1, 1)
y_test_phys = df_test['id sensitivity'].values.reshape(-1, 1)

# 2. Normalize & Train
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train_phys).ravel()
y_test_scaled = scaler.transform(y_test_phys).ravel()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_scaled)

# 3. Calculate Metrics
y_pred_scaled = rf.predict(X_test)
r2_vth = r2_score(y_test_scaled, y_pred_scaled)
mse_vth = mean_squared_error(y_test_scaled, y_pred_scaled)
rmse_vth = np.sqrt(mse_vth)
mae_vth = mean_absolute_error(y_test_scaled, y_pred_scaled)
y_pred_phys = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
acc_vth = 100 - (np.mean(np.abs((y_test_phys.ravel() - y_pred_phys) / y_test_phys.ravel())) * 100)

# Print Block
print("========================================")
print("   WIDTH VS. VTH SENSITIVITY METRICS    ")
print("========================================")
print(f"R2 Score: {r2_vth:.6f}")
print(f"MSE:      {mse_vth:.8f}")
print(f"RMSE:     {rmse_vth:.8f}")
print(f"MAE:      {mae_vth:.8f}")
print(f"Accuracy: {acc_vth:.4f}% (Physical)")
print("========================================\n")

# 4. Plotting with tolerance filter
plt.figure(figsize=(5, 4))

# Plot simulated curve (test data)
plt.plot(X_test, y_test_scaled, color='blue', label='Simulated Curve')

# Predict on training data
y_train_pred_scaled = rf.predict(X_train)

# Filter relevant training points: within test range
train_mask = (X_train.ravel() >= X_test.min()) & (X_train.ravel() <= X_test.max())
X_train_relevant = X_train[train_mask].ravel()
y_train_pred_relevant = y_train_pred_scaled[train_mask]

# ✅ Apply tolerance filter (e.g., keep points at least 0.5 units apart in Width)
tolerance = 0.05
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
            color='red', alpha=0.6, s=25, label=f'Predicted Points')

plt.title('Concave: Width vs Id Sensitivity')
plt.xlabel('Width(nm)')
plt.ylabel('Id Sensitivity')
plt.legend()
plt.grid(True)
plt.show()

