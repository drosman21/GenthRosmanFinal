import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and prepare the data
df = pd.read_csv("aa_2024_combined.csv").dropna()

non_features = ["Name", "Team", "Level", "PlayerId"]
X = df.drop(columns=non_features)
names = df["Name"].values
player_ids = df["PlayerId"].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dummy targets (replace with true Off/PA if available)
dummy_target = np.random.rand(len(X_scaled))

# Fit Ridge and XGBoost
ridge = Ridge(alpha=1.0).fit(X_scaled, dummy_target)
xgb = XGBRegressor().fit(X_scaled, dummy_target)

# Predict
ridge_preds = ridge.predict(X_scaled)
xgb_preds = xgb.predict(X_scaled)

# Create prediction DataFrame
results_df = pd.DataFrame({
    "Name": names,
    "PlayerId": player_ids,
    "Ridge_Off_per_PA": ridge_preds,
    "XGBoost_Off_per_PA": xgb_preds
})

# Get top 5 from each
top_ridge = results_df.nlargest(5, "Ridge_Off_per_PA")
top_xgb = results_df.nlargest(5, "XGBoost_Off_per_PA")

# Plot Ridge predictions
plt.figure(figsize=(8, 6))
plt.scatter(range(len(ridge_preds)), ridge_preds, alpha=0.5)
plt.scatter(top_ridge.index, top_ridge["Ridge_Off_per_PA"], color='red')
for idx, row in top_ridge.iterrows():
    plt.text(idx, row["Ridge_Off_per_PA"], row["Name"], fontsize=8)
plt.title("Ridge Regression Predicted Off/PA")
plt.xlabel("Player Index")
plt.ylabel("Predicted Off/PA")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/future_ridge.svg")
plt.close()


# Plot XGBoost predictions
plt.figure(figsize=(8, 6))
plt.scatter(range(len(xgb_preds)), xgb_preds, alpha=0.5)
plt.scatter(top_xgb.index, top_xgb["XGBoost_Off_per_PA"], color='red')
for idx, row in top_xgb.iterrows():
    plt.text(idx, row["XGBoost_Off_per_PA"], row["Name"], fontsize=8)
plt.title("XGBoost Predicted Off/PA")
plt.xlabel("Player Index")
plt.ylabel("Predicted Off/PA")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/future_xgb.svg")
plt.close()


# Print top 5 predictions
print("Top 5 Predicted Off/PA - Ridge Regression:")
print(top_ridge[["Name", "Ridge_Off_per_PA"]])

print("\nTop 5 Predicted Off/PA - XGBoost:")
print(top_xgb[["Name", "XGBoost_Off_per_PA"]])
