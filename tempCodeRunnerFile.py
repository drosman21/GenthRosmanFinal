import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# 1. Load the dataset
df = pd.read_csv("mergedfinal.csv")

# 2. Drop rows with missing or zero Off or MLB_PA
df = df.dropna(subset=["Off", "MLB_PA"])
df = df[df["MLB_PA"] > 0]

# 3. Create the modeling target
df["Off_per_PA"] = df["Off"] / df["MLB_PA"]

# 4. Drop metadata and irrelevant columns
df = df.drop(columns=["Off", "MLB_PA", "Name", "Team", "Level", "Season"], errors="ignore")

# 5. Define features and target
X_all = df.drop(columns=["Off_per_PA"])
y_all = df["Off_per_PA"]

# 6. Use a subset of players for training (e.g. 80 players)
train_df = df.sample(n=80, random_state=42)
X_train = train_df.drop(columns=["Off_per_PA"])
y_train = train_df["Off_per_PA"]

# 7. Use the entire dataset for testing
X_test = X_all
y_test = y_all

# 8. Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 9. Predict on full dataset
y_pred = model.predict(X_test)

# 10. Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# 11. Feature importances
importances = model.feature_importances_
for name, importance in sorted(zip(X_all.columns, importances), key=lambda x: -x[1]):
    print(f"{name}: {importance:.4f}")

# 12. Save predictions to CSV
results_df = X_all.copy()
results_df["Actual_Off_per_PA"] = y_test
results_df["Predicted_Off_per_PA"] = y_pred
results_df.to_csv("xgboost_off_per_pa_predictions.csv", index=False)

# 13. Save prediction plot
os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Off per PA")
plt.ylabel("Predicted Off per PA")
plt.title("XGBoost: Off/PA Prediction on Full Dataset")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/xgboost_pred_vs_actual_off_per_pa.svg")
plt.close()
