import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BIC
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

print("Dheeraj Krushna/192525230")

np.random.seed(42)
n = 500
age = np.random.choice(["Young", "Middle", "Old"], size=n, p=[0.4, 0.4, 0.2])
income = np.random.choice(["Low", "Medium", "High"], size=n, p=[0.3, 0.4, 0.3])
vehicle = np.random.choice(["Sedan", "SUV", "Truck"], size=n, p=[0.5, 0.3, 0.2])

claim_prob = 0.1 + (age == "Young") * 0.2 + (income == "Low") * 0.15 + (vehicle == "Truck") * 0.1
claim_prob = np.clip(claim_prob, 0, 1)
claim = np.array(["Yes" if np.random.rand() < p else "No" for p in claim_prob])

data = pd.DataFrame({"Age": age, "Income": income, "VehicleType": vehicle, "InsuranceClaim": claim})

hc = HillClimbSearch(data)
best_model_structure = hc.estimate(scoring_method=BIC(data))

model = DiscreteBayesianNetwork(best_model_structure.edges())
model.fit(data, estimator=DiscreteMLE())

print("Learned Bayesian Network Structure (edges):")
print(list(model.edges()))

infer = VariableElimination(model)
all_evidence = {"Age": "Young", "Income": "Low", "VehicleType": "Truck"}
model_nodes = set(model.nodes())
evidence = {k: v for k, v in all_evidence.items() if k in model_nodes and k != "InsuranceClaim"}
result = infer.query(variables=["InsuranceClaim"], evidence=evidence)
print("Evidence used:", evidence)
print(result)

'''
OUTPUT:
Dheeraj Krushna/192525230
Learned Bayesian Network Structure (edges):
[('Age', 'InsuranceClaim'), ('InsuranceClaim', 'VehicleType')]
Evidence used: {'Age': 'Young', 'VehicleType': 'Truck'}
+---------------------+------------------------+
| InsuranceClaim      |   phi(InsuranceClaim) |
+=====================+========================+
| InsuranceClaim(No)  |                 0.4405 |
+---------------------+------------------------+
| InsuranceClaim(Yes) |                 0.5595 |
+---------------------+------------------------+
'''
