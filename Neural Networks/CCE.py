# Forward pass
x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b

# ReLU activation function
y = max(z, 0)

# Backwards pass

dvalue = 1 # The derivative of the next layer.

        # Mult. the derivative with relu der
drelu_dz = dvalue * (1 if z > 0 else 0)

# D of sum of all the weights*inputs
dsum_dwx0 = 1
dsum_dwx1 = 1
dsum_dwx2 = 1
dsum_db = 1

# * Relu D with sum D
drelu_dwx0 = drelu_dz * dsum_dwx0
drelu_dwx1 = drelu_dz * dsum_dwx1
drelu_dwx2 = drelu_dz * dsum_dwx2
drelu_db = drelu_dz * dsum_db

# D of mult w.r.t x
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]

# D of mult w.r.t w
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

# Final representation of how each individual weight and input affects the ReLU output.
drelu_dw0 = drelu_dwx0 * dmul_dx0
drelu_dx0 = drelu_dwx0 * dmul_dw0
drelu_dw1 = drelu_dwx1 * dmul_dx1
drelu_dx1 = drelu_dwx1 * dmul_dw1
drelu_dw2 = drelu_dwx2 * dmul_dx2
drelu_dx2 = drelu_dwx2 * dmul_dw2
