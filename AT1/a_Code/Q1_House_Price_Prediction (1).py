import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

np.random.seed(42)

n_samples = 300
house_size_sqft = np.random.normal(1800, 600, n_samples).clip(500, 4500)
noise = np.random.normal(0, 25000, n_samples)
price = 50000 + (120 * house_size_sqft) + noise
price = price.clip(50000, None)

df = pd.DataFrame({
    "HouseSize_SqFt": house_size_sqft.round(1),
    "Price": price.round(2)
})

print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())

X = df[["HouseSize_SqFt"]]
y = df["Price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.coef_[0])
print(model.intercept_)

y_pred = model.predict(X_test)

comparison_df = pd.DataFrame({
    "Actual_Price": y_test.values[:10].round(2),
    "Predicted_Price": y_pred[:10].round(2)
})
print(comparison_df)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

with open("Q1_results.txt", "w") as f:
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Testing samples: {X_test.shape[0]}\n")
    f.write(f"Coefficient: {model.coef_[0]:.4f}\n")
    f.write(f"Intercept: {model.intercept_:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(X["HouseSize_SqFt"], y, color="steelblue", alpha=0.5, label="Actual Data")
axes[0].scatter(
    scaler.inverse_transform(X_test)[:, 0], y_test,
    color="orange", alpha=0.8, label="Test Data"
)
line_x = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)
line_y = model.predict(line_x)
axes[0].plot(
    scaler.inverse_transform(line_x)[:, 0], line_y,
    color="red", linewidth=2, label="Regression Line"
)
axes[0].set_xlabel("House Size (sq. ft.)")
axes[0].set_ylabel("Price")
axes[0].set_title("Linear Regression: House Size vs Price")
axes[0].legend()

axes[1].scatter(y_test, y_pred, color="green", alpha=0.6)
axes[1].plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
    color="red", linestyle="--", label="Ideal Fit"
)
axes[1].set_xlabel("Actual Price")
axes[1].set_ylabel("Predicted Price")
axes[1].set_title(f"Actual vs Predicted (R2 = {r2:.4f})")
axes[1].legend()

plt.tight_layout()
plt.savefig("Q1_house_price_plot.png", dpi=150)
