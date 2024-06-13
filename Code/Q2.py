import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import os

# Load the data
cwd = os.getcwd()
dataFolderPath = cwd + "\\Data"
data = pd.read_csv(dataFolderPath + "\\spine.csv")

# Split the data into features and labels
X = data.iloc[:, :-2].values
y = data.iloc[:, -2].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.thetas = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.thetas = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.epochs):
            linear_model = np.dot(X, self.thetas) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.thetas -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.thetas) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

# Train the logistic regression model from scratch
model_scratch = LogisticRegressionScratch(learning_rate=0.01, epochs=10000)
model_scratch.fit(X_train, y_train)

# Predict on test data
y_pred_scratch = model_scratch.predict(X_test)

# Evaluate the model
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
conf_matrix_scratch = confusion_matrix(y_test, y_pred_scratch)

print("Accuracy (Scratch):", accuracy_scratch)
print("Confusion Matrix (Scratch):\n", conf_matrix_scratch)

# Train the built-in logistic regression model
model_builtin = LogisticRegression(max_iter=1000000)
model_builtin.fit(X_train, y_train)

# Predict on test data
y_pred_builtin = model_builtin.predict(X_test)

# Evaluate the built-in model
accuracy_builtin = accuracy_score(y_test, y_pred_builtin)
conf_matrix_builtin = confusion_matrix(y_test, y_pred_builtin)

print("Accuracy (Built-in):", accuracy_builtin)
print("Confusion Matrix (Built-in):\n", conf_matrix_builtin)
