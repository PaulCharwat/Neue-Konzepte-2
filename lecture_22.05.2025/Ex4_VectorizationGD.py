import numpy as np
import matplotlib.pyplot as plt
import time

# Generate synthetic dataset
np.random.seed(0)
m, j = 1000, 10  # m = samples, j = features
X = np.random.rand(m, j)
beta_true = np.random.rand(j)
beta0_true = 0.5
y = X @ beta_true + beta0_true + np.random.randn(m) * 0.1  # Add noise

# Cost function
def MSE(X, y, beta, beta0):
    m = len(y)
    predictions = X @ beta + beta0
    errors = predictions - y
    return (1 / (2 * m)) * np.dot(errors, errors)

# Implement gradient descent as a loop 
def gradient_descent_loop(X, y, alpha, iterations):
    m, j = X.shape
    beta = np.zeros(j)
    beta0 = 0.0
    cost_history = []

    for it in range(iterations):
        #Initialize gradients to zero
        gradients = np.zeros(j)
        intercept_grad = 0.0
        
        #Step 1: Compute gradients
        for i in range(m):
            prediction = np.dot(X[i], beta) + beta0
            error = prediction - y[i]
            
            # Update gradients for each feature
            for j_idx in range(j):
                gradients[j_idx] += error * X[i, j_idx]
            
            # Update gradient for intercept
            intercept_grad += error
        
        # Scale gradients by 1/m
        gradients = gradients / m
        intercept_grad = intercept_grad / m
        
        #Step 2: Update parameters
        beta = beta - alpha * gradients
        beta0 = beta0 - alpha * intercept_grad
        
        #Step 3: Compute and store cost to track cost history
        cost = MSE(X, y, beta, beta0)
        cost_history.append(cost)
        
    return beta, beta0, cost_history

#Implement vectorized gradient descent
def gradient_descent_vectorized(X, y, alpha, iterations):
    m, j = X.shape
    beta = np.zeros(j)
    beta0 = 0.0
    cost_history = []

    for it in range(iterations):
        predictions = X @ beta + beta0
        error = predictions - y
        
        #Step 1: Compute gradients
        gradients = (1/m) * (X.T @ error)  # Vectorized gradient for beta
        intercept_grad = (1/m) * np.sum(error)  # Gradient for beta0
        
        #Step 2: Update parameters
        beta = beta - alpha * gradients
        beta0 = beta0 - alpha * intercept_grad
        
        #Step 3: Compute and store cost to track cost history
        cost = MSE(X, y, beta, beta0)
        cost_history.append(cost)
        
    return beta, beta0, cost_history

# Run and compare
alpha = 0.1
iterations = 100

start = time.time()
beta_loop, beta0_loop, cost_loop = gradient_descent_loop(X, y, alpha, iterations)
end = time.time()
print(f"Loop version time: {end - start:.4f}s")

start = time.time()
beta_vec, beta0_vec, cost_vec = gradient_descent_vectorized(X, y, alpha, iterations)
end = time.time()
print(f"Vectorized version time: {end - start:.4f}s")

# Plotting
plt.plot(cost_loop, label="Loop-based")
plt.plot(cost_vec, label="Vectorized", linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Iterations")
plt.legend()
plt.grid(True)
plt.show()