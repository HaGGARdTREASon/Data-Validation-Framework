import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import argparse

# ─────────────────────────────────────────────
# CLI argument: --train_ratio (default = 0.8)
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Convex Width vs Id Sensitivity — configurable train/test split"
)
parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.5,
    help="Fraction of data used for training (e.g. 0.6 → 60%% train, 40%% test). Default: 0.8",
)
args = parser.parse_args()

train_ratio = args.train_ratio
if not (0.0 < train_ratio < 1.0):
    raise ValueError("--train_ratio must be a float strictly between 0 and 1.")
test_ratio = 1.0 - train_ratio

print(f"\n  Split → Train: {train_ratio*100:.0f}%  |  Test: {test_ratio*100:.0f}%\n")

# ─────────────────────────────────────────────
# 1. Load the single dataset
# ─────────────────────────────────────────────
data_file = "dataset convex_8000_nw.xlsx"   # ← change if your file name differs

df = pd.read_excel(data_file).dropna(subset=["Width", "id sensitivity"])
df = df.sort_values(by="Width").reset_index(drop=True)

X = df[["Width"]].values
y = df["id sensitivity"].values.reshape(-1, 1)

# ─────────────────────────────────────────────
# 2. Split
# ─────────────────────────────────────────────
X_train, X_test, y_train_phys, y_test_phys = train_test_split(
    X, y,
    test_size=test_ratio,
    random_state=42,
    shuffle=True,          # shuffle before splitting
)

# Sort test set by Width so the curve looks clean on the plot
sort_idx = X_test.ravel().argsort()
X_test = X_test[sort_idx]
y_test_phys = y_test_phys[sort_idx]

# ─────────────────────────────────────────────
# 3. Normalize & Train
# ─────────────────────────────────────────────
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train_phys).ravel()
y_test_scaled  = scaler.transform(y_test_phys).ravel()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_scaled)

# ─────────────────────────────────────────────
# 4. Metrics
# ─────────────────────────────────────────────
y_pred_scaled = rf.predict(X_test)
r2_id   = r2_score(y_test_scaled, y_pred_scaled)
mse_id  = mean_squared_error(y_test_scaled, y_pred_scaled)
rmse_id = np.sqrt(mse_id)
mae_id  = mean_absolute_error(y_test_scaled, y_pred_scaled)

y_pred_phys = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
acc_id = 100 - (
    np.mean(np.abs((y_test_phys.ravel() - y_pred_phys) / y_test_phys.ravel())) * 100
)

print("========================================")
print("    WIDTH VS. ID SENSITIVITY METRICS    ")
print(f"  (Train {train_ratio*100:.0f}% / Test {test_ratio*100:.0f}%)")
print("========================================")
print(f"R2 Score: {r2_id:.6f}")
print(f"MSE:      {mse_id:.8f}")
print(f"RMSE:     {rmse_id:.8f}")
print(f"MAE:      {mae_id:.8f}")
print(f"Accuracy: {acc_id:.4f}% (Physical)")
print("========================================\n")

# ─────────────────────────────────────────────
# 5. Plot
# ─────────────────────────────────────────────
plt.figure(figsize=(5, 4))

# Simulated/true test curve
plt.plot(
    X_test, y_test_scaled,
    color="blue", linewidth=2, zorder=1,
    label=f"Simulated Curve"
)

# Predicted points on training data (filtered for density)
y_train_pred_scaled = rf.predict(X_train)

train_mask = (
    (X_train.ravel() >= X_test.min()) &
    (X_train.ravel() <= X_test.max())
)
X_tr_rel = X_train[train_mask].ravel()
y_tr_rel  = y_train_pred_scaled[train_mask]

# Sort for cleaner density filter
sort_tr = X_tr_rel.argsort()
X_tr_rel, y_tr_rel = X_tr_rel[sort_tr], y_tr_rel[sort_tr]

tolerance = 0.08
X_filtered, y_filtered = [], []
last_x = -np.inf
for x, y_val in zip(X_tr_rel, y_tr_rel):
    if x - last_x >= tolerance:
        X_filtered.append(x)
        y_filtered.append(y_val)
        last_x = x

plt.scatter(
    X_filtered, y_filtered,
    color="red", alpha=0.6, s=25, zorder=2,
    label=f"Predicted Points"
)

plt.xlabel("Width(nm)")
plt.ylabel("Id Sensitivity")
plt.title(
    f"Convex: Width vs Id Sensitivity"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()