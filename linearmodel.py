import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# load datasets
train_df = pd.read_csv("mergedfinal.csv")
test_df = pd.read_csv("testset.csv")

# drop rows with missing or zero Off or MLB_PA
train_df = train_df.dropna(subset=["Off", "MLB_PA"])
train_df = train_df[train_df["MLB_PA"] > 0]
test_df = test_df.dropna(subset=["Off", "MLB_PA"])
test_df = test_df[test_df["MLB_PA"] > 0]

#create target column
train_df["Off_per_PA"] = train_df["Off"] / train_df["MLB_PA"]
test_df["Off_per_PA"] = test_df["Off"] / test_df["MLB_PA"]

# drop irrelevant columns
drop_cols = ["Off", "MLB_PA", "Name", "Team", "Level", "Season"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df = test_df.drop(columns=drop_cols, errors="ignore")

# define features and targets
X_train = train_df.drop(columns=["Off_per_PA"])
y_train = train_df["Off_per_PA"]
X_test = test_df.drop(columns=["Off_per_PA"])
X_test = X_test[X_train.columns]  # Align columns
y_test = test_df["Off_per_PA"]

#train Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# predict on test and training sets
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# evaluate performance
def evaluate(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))

evaluate("Test Set", y_test, y_pred_test)
evaluate("Training Set", y_train, y_pred_train)

# feature importance
coefs = model.coef_
for name, coef in sorted(zip(X_train.columns, coefs), key=lambda x: abs(x[1]), reverse=True):
    print(f"{name}: {coef:.4f}")

#save predictions and plots
os.makedirs("figures", exist_ok=True)

# test set results
results_test = X_test.copy()
results_test["Actual_Off_per_PA"] = y_test
results_test["Predicted_Off_per_PA"] = y_pred_test
results_test.to_csv("ridge_predictions_on_testset.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7)
plt.xlabel("Actual Off per PA")
plt.ylabel("Predicted Off per PA")
plt.title("Ridge Regression: Off/PA Prediction on Test Set")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/ridge_pred_vs_actual_testset.svg")
plt.close()

# training set results
results_train = X_train.copy()
results_train["Actual_Off_per_PA"] = y_train
results_train["Predicted_Off_per_PA"] = y_pred_train
results_train.to_csv("ridge_predictions_on_trainset.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_train, alpha=0.7)
plt.xlabel("Actual Off per PA")
plt.ylabel("Predicted Off per PA")
plt.title("Ridge Regression: Off/PA Prediction on Training Set")
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/ridge_pred_vs_actual_trainset.svg")
plt.close()


