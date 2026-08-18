import numpy as np
from hmmlearn import hmm

print("Dheeraj Krushna/192525230")

states = ["Sunny", "Cloudy", "Rainy"]
observations = ["Sunny", "Cloudy", "Rainy"]

model = hmm.CategoricalHMM(n_components=3, random_state=42)
model.startprob_ = np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
model.emissionprob_ = np.array([
    [0.8, 0.15, 0.05],
    [0.3, 0.5, 0.2],
    [0.1, 0.3, 0.6]
])

obs_sequence = np.array([[0, 0, 2, 1, 2]]).T
logprob, hidden_states = model.decode(obs_sequence, algorithm="viterbi")

observed = [observations[i] for i in obs_sequence.flatten()]
predicted = [states[i] for i in hidden_states]

print("Observed sequence:", observed)
print("Predicted hidden states:", predicted)
print("Log Probability:", logprob)

'''
OUTPUT:
Dheeraj Krushna/192525230
Observed sequence: ['Sunny', 'Sunny', 'Rainy', 'Cloudy', 'Rainy']
Predicted hidden states: ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Rainy']
Log Probability: -7.228291176304996
'''
