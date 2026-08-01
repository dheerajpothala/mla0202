import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [1000, 2],
    [1500, 3],
    [800, 2],
    [1200, 3],
    [2000, 4]
])
y = np.array([50, 75, 40, 60, 90])

model = LinearRegression()
model.fit(X, y)

b0 = model.intercept_
b1 = model.coef_[0]
b2 = model.coef_[1]

print("(i) Regression Model:")
print(f"Price = {b0:.4f} + ({b1:.4f})*Area + ({b2:.4f})*Bedrooms")

print("\n(ii) Estimated Regression Coefficients:")
print("Intercept (b0):", b0)
print("Coefficient for Area (b1):", b1)
print("Coefficient for Bedrooms (b2):", b2)

pred = model.predict(X)
print("\nPredicted Prices:", pred)
print("R2 Score:", model.score(X, y))

print("\n(iii) Interpretation of Coefficients:")
print(f"b0 = {b0:.4f}: Predicted price when Area=0 and Bedrooms=0 (baseline, not practically meaningful).")
print(f"b1 = {b1:.4f}: Price increases by {b1:.4f} lakhs for every 1 sq.ft increase in Area, holding Bedrooms constant.")
print(f"b2 = {b2:.4f}: Price changes by {b2:.4f} lakhs for every additional Bedroom, holding Area constant.")

"""
Output:
(i) Regression Model:
Price = 8.5047 + (0.0425)*Area + (-0.2804)*Bedrooms

(ii) Estimated Regression Coefficients:
Intercept (b0): 8.50467289719628
Coefficient for Area (b1): 0.0425233644859813
Coefficient for Bedrooms (b2): -0.2803738317757003

Predicted Prices: [50.46728972 71.44859813 41.96261682 58.69158879 92.42990654]
R2 Score: 0.9846208446705312

(iii) Interpretation of Coefficients:
b0 = 8.5047: Predicted price when Area=0 and Bedrooms=0 (baseline, not practically meaningful).
b1 = 0.0425: Price increases by 0.0425 lakhs for every 1 sq.ft increase in Area, holding Bedrooms constant.
b2 = -0.2804: Price changes by -0.2804 lakhs for every additional Bedroom, holding Area constant.
"""
