import numpy as np

# Incoming gradients
dvalues = np.array([[1., 1., 1.]])

# Weights: 4 inputs, 3 neurons
weights = np.array([
    [0.2, 0.5, -0.26],
    [0.8, -0.91, -0.27],
    [-0.5, 0.26, 0.17],
    [1.0, -0.5, 0.87]
])

# Backpropagate:
x = np.dot(dvalues, weights.T)

print(x)