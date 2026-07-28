import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import the required algorithms
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# 1. Load Data
train_file = "dataset convex_8000_nw.xlsx"
test_file = "Convex curve-testing.xlsx"

try:
    df_train = pd.read_excel(train_file).dropna(subset=['Width', 'id sensitivity'])
    df_test = pd.read_excel(test_file).dropna(subset=['Width', 'id sensitivity']).sort_values(by='Width')
except FileNotFoundError:
    print(f"Error: Could not find the dataset files. Please ensure '{train_file}' and '{test_file}' are in the same directory.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    sys.exit(1)

X_train, X_test = df_train[['Width']].values, df_test[['Width']].values
y_train_phys = df_train['id sensitivity'].values.reshape(-1, 1)
y_test_phys = df_test['id sensitivity'].values.reshape(-1, 1)

# 2. Normalize 
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train_phys).ravel()
y_test_scaled = scaler.transform(y_test_phys).ravel()

# Define the models to evaluate
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GBR': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBR': XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.01)
}

results = []
predictions = {}

# 3. Train, Predict & Calculate Metrics for each model
for name, model in models.items():
    # Train
    model.fit(X_train, y_train_scaled)
    
    # Predict
    y_pred_scaled = model.predict(X_test)
    predictions[name] = y_pred_scaled 
    
    # Calculate Metrics
    r2_id = r2_score(y_test_scaled, y_pred_scaled)
    mse_id = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse_id = np.sqrt(mse_id)
    mae_id = mean_absolute_error(y_test_scaled, y_pred_scaled)
    
    # Calculate Physical Accuracy
    y_pred_phys = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Avoid division by zero in accuracy calculation
    safe_y_test = np.where(y_test_phys.ravel() == 0, 1e-10, y_test_phys.ravel())
    acc_id = 100 - (np.mean(np.abs((safe_y_test - y_pred_phys) / safe_y_test)) * 100)
    
    # Store results
    results.append({
        'Algorithm': name,
        'R2 Score': r2_id,
        'MSE': mse_id,
        'RMSE': rmse_id,
        'MAE': mae_id,
        'Accuracy (%)': acc_id
    })

# 4. Print EXACT Markdown Table for comparison_results.md
print("\n# Algorithm Comparison Results\n")
print("| Algorithm | R2 Score | MSE | RMSE | **MAE** | Accuracy (%) |")
print("| ----- | ----- | ----- | ----- | ----- | ----- |")
for res in results:
    print(f"| **{res['Algorithm']}** | {res['R2 Score']:.6f} | {res['MSE']:.6f} | {res['RMSE']:.6f} | **{res['MAE']:.6f}** | {res['Accuracy (%)']:.6f} |")
print("\n")

# 5. Plot True Curve vs All Predictions
plt.figure(figsize=(12, 7))

# Plot simulated curve (true test data)
plt.plot(X_test, y_test_scaled, color='black', label='True Curve (Test)', linewidth=3, zorder=10)

# Plot predictions for each algorithm
colors = ['red', 'blue', 'green', 'orange', 'purple']
for (name, y_pred), color in zip(predictions.items(), colors):
    plt.plot(X_test, y_pred, linestyle='--', color=color, label=f'{name} Prediction', alpha=0.7)

plt.xlabel('Width')
plt.ylabel('Id Sensitivity (Scaled)')
plt.title('Algorithm Comparison: Convex Width vs Id Sensitivity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()