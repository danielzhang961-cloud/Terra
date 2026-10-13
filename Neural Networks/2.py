import numpy as np

# Inputs, weights and bias
x = np.array([1.0, -2.0, 3.0])
w = np.array([-3.0, -1.0, 2.0])
b = 1.0

# Forward pass
z = np.dot(x, w) + b
y = max(0, z)

print(y)  # 6.0

# Incoming gradient
dvalue = 1.0

# ReLU derivative + chain rule
drelu = dvalue * (1.0 if z > 0 else 0.0)

# Gradients
dx = drelu * w
dw = drelu * x
db = drelu

print("Input gradients:", dx)
print("Weight gradients:", dw)
print("Bias gradient:", db)

learning_rate = 0.001

w -= learning_rate * dw
b -= learning_rate * db

z = np.dot(x, w) + b
y = max(0, z)
print()
print("Updated Weights:", w)
print("Updated Bias gradient:", b)

print("Updated output:", y)

incoming_gradients = w