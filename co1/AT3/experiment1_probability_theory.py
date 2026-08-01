import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
names = {0: "Malignant", 1: "Benign"}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

prior = {0: (y_train == 0).mean(), 1: (y_train == 1).mean()}
stats = {c: {"mean": X_train[y_train == c].mean(), "var": X_train[y_train == c].var()} for c in [0, 1]}

def likelihood(x, mean, var):
    return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

def predict(row):
    scores = {c: np.log(prior[c]) + np.sum(np.log(likelihood(row, stats[c]["mean"], stats[c]["var"]) + 1e-300)) for c in [0, 1]}
    return max(scores, key=scores.get), scores

pred, scores = predict(X_test.iloc[0])
print(f"P(Malignant)={prior[0]:.4f}  P(Benign)={prior[1]:.4f}")
print(f"Predicted={names[pred]}  Actual={names[y_test.iloc[0]]}")
print(f"Score(Malignant)={scores[0]:.4f}  Score(Benign)={scores[1]:.4f}")

y_pred = X_test.apply(lambda r: predict(r)[0], axis=1)
print(f"Accuracy={accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))
