import numpy as np
from sklearn.naive_bayes import CategoricalNB

X = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])
y = np.array(["Yes", "Yes", "No", "Yes", "No"])

model = CategoricalNB()
model.fit(X, y)

classes = model.classes_
prior = np.exp(model.class_log_prior_)

print("(i) Prior Probabilities:")
for c, p in zip(classes, prior):
    print(f"P(Disease={c}) = {p:.4f}")

print("\n(ii) Conditional Probabilities:")
for i, feature in enumerate(["Fever", "Headache"]):
    for c_idx, c in enumerate(classes):
        probs = np.exp(model.feature_log_prob_[i][c_idx])
        for val, p in zip([0, 1], probs):
            print(f"P({feature}={val}|Disease={c}) = {p:.4f}")

test = np.array([[1, 0]])
pred = model.predict(test)
prob = model.predict_proba(test)

print("\n(iii) Prediction for Fever=1, Headache=0:")
print("Predicted Disease:", pred[0])
print(f"Probability [No, Yes]: {prob}")