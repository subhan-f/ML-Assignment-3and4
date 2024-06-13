import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def cost_function(X, Y, theta):
    m = len(Y)
    J = np.sum((X.dot(theta) - Y) ** 2)/(2 * m)
    return J

def gradient_descent(X, Y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    m = len(Y)
    for iteration in range(iterations):
        h_theta = X.dot(theta) # Model value
        loss = h_theta - Y # Difference b/w model and actual Y
        gradient = X.T.dot(loss) / m # All partial derivatives in one line
        theta = theta - alpha * gradient # Updating theta
        cost = cost_function(X, Y, theta) # New cost value
        cost_history[iteration] = cost
    return theta, cost_history


cwd = os.getcwd()
dataFolderPath = cwd + "\\Data"

data = pd.read_csv(dataFolderPath + "\\posal.csv")

print(data.shape)
print(data)
# data.head()

level = data['Level'].values
sal = data['Salary'].values

plt.scatter(level, sal)
plt.title("pos-sal graph")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()

x1 = data['Level'].values
x2 = x1**2
x3 = x1**3
x4 = x1**4

x1s=x1/np.max(x1)
x2s=x2/np.max(x2)
x3s = x3/np.max(x3)
x4s = x4/np.max(x4)

Y = data['Salary'].values

m = len(x1)

x0 = np.ones(m)

X = np.array([x0,x1s,x2s,x3s,x4s]).T

theta = np.array([0, 0, 0, 0, 0])


# learning rate
alpha = 0.0001
inital_cost = cost_function(X, Y, theta)
print("Initial Cost: ", inital_cost)

# 1000000 Iterations
finaltheta, cost_history = gradient_descent(X, Y, theta, alpha, 1000000)

print("Final Coefficients", finaltheta) # final model parameters theta
print("Final Cost", cost_history[-1]) # final cost


h_theta = X.dot(finaltheta)  # Final Model
plt.scatter(x1, sal)
plt.plot(x1,h_theta)
plt.title("pos-sal graph")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()


s=6.5
sample = np.array([1,s/np.max(x1),s**2/np.max(x2),s**3/np.max(x3),s**4/np.max(x4)])
predsal = sample.dot(finaltheta)
print(predsal)
