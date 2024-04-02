import numpy as np

def sigmoid(z):
    # add the sigmoid function
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta -= alpha * gradient
        cost_history[i] = cost_function(X, y, theta)

    return theta, cost_history

# Example data (mooring times and ship types)
X = np.array([[1.2, 3.4], [2.5, 4.7], [3.1, 5.2], [1.8, 4.0], [2.9, 5.1]])
y = np.array([1, 2, 1, 3, 2])  # Assuming ship types are represented as integers

# Add intercept term to X
X = np.c_[np.ones((X.shape[0], 1)), X]

# Initialize theta (weights)
theta = np.zeros(X.shape[1])

# Hyperparameters
alpha = 0.01
num_iters = 1000

# Perform gradient descent
theta, cost_history = gradient_descent(X, y, theta, alpha, num_iters)

# Make predictions
predictions = np.round(sigmoid(np.dot(X, theta)))

# Calculate accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)
