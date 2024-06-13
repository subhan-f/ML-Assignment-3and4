import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, pinv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# Function for cost calculation
def cost_function(X, Y, theta):
    m = len(Y)
    J = np.sum((X.dot(theta) - Y) ** 2) / (2 * m)
    return J

# Load the position level vs salary data
cwd = os.getcwd()
dataFolderPath = cwd + "\\Data"
data = pd.read_csv(dataFolderPath + "\\posal.csv")

# Extract features and labels
level = data['Level'].values
sal = data['Salary'].values

# Normalize and create polynomial features
x1 = level
x2 = x1**2
x3 = x1**3
x4 = x1**4

x1s = x1 / np.max(x1)
x2s = x2 / np.max(x2)
x3s = x3 / np.max(x3)
x4s = x4 / np.max(x4)

Y = sal
m = len(x1)
x0 = np.ones(m)
X = np.array([x0, x1s, x2s, x3s, x4s]).T

# Using normal equations to calculate theta
theta_normal_eq = inv(X.T.dot(X)).dot(X.T).dot(Y)

# Plotting the results from normal equations
plt.scatter(level, sal, color='red')
plt.plot(level, X.dot(theta_normal_eq), color='blue')
plt.title('Position level vs Salary (Normal Equations)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Using sklearn LinearRegression for comparison
poly_features = np.vstack((x1s, x2s, x3s, x4s)).T
lin_reg = LinearRegression()
lin_reg.fit(poly_features, Y)
theta_sklearn = np.hstack(([lin_reg.intercept_], lin_reg.coef_))

# Predict a specific level using both models for comparison
s = 6.5
sample = np.array([1, s / np.max(x1), s**2 / np.max(x2), s**3 / np.max(x3), s**4 / np.max(x4)])
pred_sal_normal_eq = sample.dot(theta_normal_eq)
pred_sal_sklearn = sample.dot(theta_sklearn)

print("Prediction using normal equations:", pred_sal_normal_eq)
print("Prediction using sklearn LinearRegression:", pred_sal_sklearn)
print("Normal Equations Coefficients:", theta_normal_eq)
print("Sklearn Coefficients:", theta_sklearn)

# Load the admit dataset
admit_data = pd.read_csv(dataFolderPath + "\\admit.csv")

# Extract features and labels
X_admit = admit_data[['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research']]
Y_admit = admit_data['Chance of Admit']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_admit, Y_admit, test_size=0.2, random_state=42)

# Using normal equations for the admission dataset
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # add x0 = 1 to each instance
theta_admit = pinv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(Y_train)

X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]  # add x0 = 1 to each instance
Y_pred_normal_eq = X_test_b.dot(theta_admit)

# Using sklearn LinearRegression for the admission dataset
lin_reg_admit = LinearRegression()
lin_reg_admit.fit(X_train, Y_train)
Y_pred_sklearn = lin_reg_admit.predict(X_test)

# Report accuracy using r2_score
r2_normal_eq = r2_score(Y_test, Y_pred_normal_eq)
r2_sklearn = r2_score(Y_test, Y_pred_sklearn)

print("R2 score using normal equations:", r2_normal_eq)
print("R2 score using sklearn LinearRegression:", r2_sklearn)
