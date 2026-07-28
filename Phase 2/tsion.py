import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# 1. Define file names
TRAIN_FILE = "dataset tsi_7000.xlsx"
TEST_FILE = "tsi_validation.xlsx"

SHEET_NAME_TRAIN = "dataset tsi_7000"
SHEET_NAME_TEST = "Sheet1" 

print("--- Loading Data ---")
if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
    print("Error: Files not found.")
    exit()

# Load datasets
df_train = pd.read_excel(TRAIN_FILE, sheet_name=SHEET_NAME_TRAIN)
df_test = pd.read_excel(TEST_FILE, sheet_name=SHEET_NAME_TEST)

# Strip spaces from column names to prevent KeyError
df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

# Dynamically find the Ion sensitivity column names (handles 'Ion_sensitivity' or 'Ion sensitivity')
train_ion_col = [c for c in df_train.columns if 'ion' in c.lower() and 'sens' in c.lower()][0]
test_ion_col = [c for c in df_test.columns if 'ion' in c.lower() and 'sens' in c.lower()][0]

df_train = df_train.dropna(subset=['Tsi', train_ion_col])
df_test = df_test.dropna(subset=['Tsi', test_ion_col]).sort_values(by='Tsi')

X_train, X_test = df_train[['Tsi']].values, df_test[['Tsi']].values

# Extract values as flat 1D arrays for standard training
y_train_phys = df_train[train_ion_col].values.ravel()
y_test_phys = df_test[test_ion_col].values.ravel()

# 2. Train Model directly on Physical Values (No Normalization)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_phys)

# 3. Calculate Metrics directly on the physical scale
y_pred_phys = rf.predict(X_test)

r2_ion = r2_score(y_test_phys, y_pred_phys)
mse_ion = mean_squared_error(y_test_phys, y_pred_phys)
rmse_ion = np.sqrt(mse_ion)
mae_ion = mean_absolute_error(y_test_phys, y_pred_phys)

# Calculate physical percentage Accuracy
acc_ion = 100 - (np.mean(np.abs((y_test_phys - y_pred_phys) / y_test_phys)) * 100)

print("========================================")
print("     TSI VS. ION SENSITIVITY METRICS    ")
print("========================================")
print(f"R2 Score: {r2_ion:.6f}")
print(f"MSE:      {mse_ion:.8e}")
print(f"RMSE:     {rmse_ion:.8e}")
print(f"MAE:      {mae_ion:.8e}")
print(f"Accuracy: {acc_ion:.4f}%")
print("========================================\n")

# 4. Plot Training Predictions vs Test Curve (USING PHYSICAL VALUES)
plt.figure(figsize=(10, 6))

# A. Plot simulated curve (test data) using ORIGINAL PHYSICAL VALUES
plt.plot(X_test, y_test_phys, color='orange', label='Simulated Curve (Test)', linewidth=2.5, zorder=1)

# B. Predict on training data (Native physical predictions)
y_train_pred_phys = rf.predict(X_train)

# Filter relevant training points: within test range
train_mask = (X_train.ravel() >= X_test.min()) & (X_train.ravel() <= X_test.max())
X_train_relevant = X_train[train_mask].ravel()
y_train_pred_relevant = y_train_pred_phys[train_mask] 

# Sort the points before spacing them out to ensure they are parsed left-to-right
sort_idx = np.argsort(X_train_relevant)
X_train_relevant = X_train_relevant[sort_idx]
y_train_pred_relevant = y_train_pred_relevant[sort_idx]

# C. Apply user-defined tolerance filter (0.0099 minimum X-axis spacing)
tolerance = 0.0099
X_filtered = []
y_filtered = []

last_x = -np.inf
for x, y in zip(X_train_relevant, y_train_pred_relevant):
    if x - last_x >= tolerance:
        X_filtered.append(x)
        y_filtered.append(y)
        last_x = x

# D. Plot filtered predicted physical points
plt.scatter(X_filtered, y_filtered, 
            color='purple', alpha=0.8, s=35, label='Predicted Points', edgecolors='black', linewidth=0.5, zorder=2)

plt.xlabel('Tsi')
plt.ylabel('Ion Sensitivity')
plt.title('Tsi vs Ion Sensitivity')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()