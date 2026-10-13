import numpy as np

# Example Dataset
def spiral_data(points, classes):
    x = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        x[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return x, y
X, y = spiral_data(points=100, classes=3)

np.random.seed(0)

# Architecture for neuron maths (weights * inputs) + biases
    # Inputs must be the no. of features in the data (2 for coords x, y), neurons can be anything
class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # Randomly generated weights to start
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Activation Functions
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True)) # Exponential Values of a single batch (list)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Loss method
class loss:
                        # Output of network, correct confidences
    def calculate(self, output, y):
        sample_losses = self.forward(output, y) # Varies based on which type of loss used
        data_loss = np.mean(sample_losses)
        # Returns batch (neuron) loss
        return data_loss

# Loss Function
class Loss_CCE(loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7 ) # Preventing infinities

        if len(y_true.shape) == 1: # If scalar value passed
                                                # List index, number index (in prediction matrix)
            correct_confidences = y_pred_clipped[range(samples), y_true] # Which index is 'correct'
        elif len(y_true.shape) == 2: # 1-hot encoded vector (2D array)
                                        # Everything gets * 0 except the 'correct' index confidence
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        # Returns individual loss
        return negative_log_likelihoods
    
# Layer 1
dense1 = layer_dense(2, 3)
activation1 = ReLU()

# Layer 2
dense2 = layer_dense(3, 3)
activation2 = Softmax()

# Passing raw data into Layer 1
dense1.forward(X)
activation1.forward(dense1.output)

# Layer 1 output into Layer 2
dense2.forward(activation1.output)

# Probability predictions for each 'class' in data (1, 2, 3) FOR every datapoint (300 of them)
activation2.forward(dense2.output)

# Calculating average loss
loss_function = Loss_CCE()
loss = loss_function.calculate(activation2.output, y)

# Accuracy
    # Prediction of model (highest confidence)
predictions = np.argmax(activation2.output, axis=1)
    # If prediction == correct answer...
accuracy = np.mean(predictions == y)

print(activation2.output[:5])
print("Loss:", loss)
print('acc:', accuracy)
