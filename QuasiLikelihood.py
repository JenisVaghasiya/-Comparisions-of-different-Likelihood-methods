#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import statsmodels.api as sm

# Step 1: Formulate a Model
np.random.seed(42)
X = sm.add_constant(np.random.random(100))  # Independent variable with a constant term
true_beta = np.array([2.0, 3.0])  # True coefficients
error_scale = 0.5 * (1 + X[:, 1])  # Heteroscedastic errors
y = X @ true_beta + np.random.normal(scale=error_scale)  # Simulated data

# Step 2: Construct a Quasi-Likelihood Function
def quasi_likelihood(y, X, params, scale):
    residuals = y - X @ params
    return -0.5 * np.sum((residuals / scale) ** 2)

# Step 3: Maximize the Quasi-Likelihood
initial_params = np.zeros(X.shape[1])  # Initial parameter values
result = sm.optimize.minimize(quasi_likelihood, initial_params, args=(X, y, error_scale))

# Extract estimated parameters
quasi_likelihood_params = result.x

# Step 4: Inference
# Conduct hypothesis tests, calculate confidence intervals, etc.
# For simplicity, we'll print the estimated parameters
print("Estimated Parameters using Quasi-Likelihood:")
print(quasi_likelihood_params)

# Compare with OLS (for illustration purposes)
ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()
print("\nOLS Estimated Parameters for Comparison:")
print(ols_results.params)


# In[2]:


pip install numpy statsmodels scipy matplotlib


# In[4]:


import matplotlib.pyplot as plt

# Scatter plot of the simulated data
plt.scatter(X[:, 1], y, label='Simulated Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter Plot of Simulated Data')
plt.legend()
plt.show()

# Plot the fitted regression line
y_pred = X @ quasi_likelihood_params
plt.scatter(X[:, 1], y, label='Simulated Data')
plt.plot(X[:, 1], y_pred, color='red', label='Quasi-Likelihood Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Quasi-Likelihood Regression Fit')
plt.legend()
plt.show()

# Display the model summary
print("Quasi-Likelihood Model Summary:")
print(result)


# In[ ]:




