import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [3, 2],
    [2, 1],
    [1, 0],
    [3, 3],
    [0, 1]
])
y = np.array([1, 1, 0, 1, 0])

model = LogisticRegression()
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]

print("(i) Linear Classification Model:")
print(f"z = ({w[0]:.4f})*Offer + ({w[1]:.4f})*Win + ({b:.4f})")
print("Spam predicted if sigmoid(z) >= 0.5, else Not Spam")

print("\n(ii) Decision Boundary:")
print(f"({w[0]:.4f})*Offer + ({w[1]:.4f})*Win + ({b:.4f}) = 0")

new_email = np.array([[2, 1]])
prediction = model.predict(new_email)
probability = model.predict_proba(new_email)

print("\n(iii) Classification of new email (Offer=2, Win=1):")
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")
print("Probability [Not Spam, Spam]:", probability)
"""
OUTPUT:

(i) Linear Classification Model:
z = (0.9278)*Offer + (0.5431)*Win + (-1.7758)
Spam predicted if sigmoid(z) >= 0.5, else Not Spam

(ii) Decision Boundary:
(0.9278)*Offer + (0.5431)*Win + (-1.7758) = 0

(iii) Classification of new email (Offer=2, Win=1):
Prediction: Spam
Probability [Not Spam, Spam]: [[0.34910736 0.65089264]]
"""
