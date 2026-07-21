import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

np.random.seed(7)

n_samples = 300
study_hours = np.random.uniform(0, 10, n_samples)
attendance = np.random.uniform(50, 100, n_samples)
internal_marks = np.random.uniform(10, 50, n_samples)
assignment_scores = np.random.uniform(5, 25, n_samples)
noise = np.random.normal(0, 4, n_samples)

final_marks = (
    3.2 * study_hours +
    0.35 * attendance +
    0.9 * internal_marks +
    0.6 * assignment_scores +
    noise - 10
)
final_marks = final_marks.clip(0, 100)

df = pd.DataFrame({
    "Study_Hours": study_hours.round(2),
    "Attendance_Percent": attendance.round(2),
    "Internal_Marks": internal_marks.round(2),
    "Assignment_Score": assignment_scores.round(2),
    "Final_Exam_Marks": final_marks.round(2)
})

print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())

feature_cols = ["Study_Hours", "Attendance_Percent", "Internal_Marks", "Assignment_Score"]
X = df[feature_cols]
y = df["Final_Exam_Marks"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_
})
print(coef_df)
print(model.intercept_)

y_pred = model.predict(X_test)

comparison_df = pd.DataFrame({
    "Actual_Marks": y_test.values[:10].round(2),
    "Predicted_Marks": y_pred[:10].round(2)
})
print(comparison_df)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

with open("Q2_results.txt", "w") as f:
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Testing samples: {X_test.shape[0]}\n")
    for feat, c in zip(feature_cols, model.coef_):
        f.write(f"{feat}: {c:.4f}\n")
    f.write(f"Intercept: {model.intercept_:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, y_pred, color="teal", alpha=0.6)
axes[0].plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
    color="red", linestyle="--", label="Ideal Fit"
)
axes[0].set_xlabel("Actual Final Marks")
axes[0].set_ylabel("Predicted Final Marks")
axes[0].set_title(f"Actual vs Predicted (R2 = {r2:.4f})")
axes[0].legend()

axes[1].barh(feature_cols, model.coef_, color="slateblue")
axes[1].set_xlabel("Coefficient (Standardized)")
axes[1].set_title("Feature Influence on Final Marks")

plt.tight_layout()
plt.savefig("Q2_student_performance_plot.png", dpi=150)
