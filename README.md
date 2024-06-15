
# Assignments 3 and 4: Neural Network Classifier for Spine Condition Classification

## Overview
This repository contains the implementation for two assignments aimed at classifying human spine conditions as normal or abnormal using the `spine.csv` dataset.

## Assignment 3: Logistic Regression Classifier

### Introduction
This assignment involves using logistic regression to classify human spine conditions as normal or abnormal using the `spine.csv` dataset. The dataset contains 12 extracted image features, including various angles, tilts, slopes, and radii, along with class labels (0 for normal, 1 for abnormal).

### Libraries Used
- `pandas` for data manipulation.
- `numpy` for array operations.
- `scikit-learn` for implementing logistic regression and evaluating model performance.

### Steps
1. **Data Loading and Preprocessing:**
   - Loaded the dataset using pandas.
   - Split the data into features (`X`) and labels (`y`).

2. **Data Splitting:**
   - Split the dataset into training and testing sets with an 80-20 split.

3. **Model Training and Evaluation:**
   - Trained a logistic regression model.
   - Evaluated the model using accuracy and a confusion matrix.

### Results
- **Accuracy:** The logistic regression model achieved an accuracy of 84.62%.

### Files
- Implementation Code: [Q3.py](Code/Q3.py)
- Output Confusion Matrix: [output-Q3.png](Code/output-Q3.png)
- Detailed Report: [ML-Assignment3.pdf](ML-Assignment3.pdf)

## Assignment 4: Fully Connected Neural Network Classifier

### Introduction
This assignment involves training a fully connected neural network to classify human spine conditions as normal or abnormal using the `spine.csv` dataset. The task was to predict the class label based on the provided features.

### Libraries Used
- `pandas` for data manipulation.
- `numpy` for array operations.
- `tensorflow` and `keras` for building and training the neural network.
- `scikit-learn` for evaluating the model performance using a confusion matrix and comparing with logistic regression and MLPClassifier.

### Steps
1. **Data Loading and Preprocessing:**
   - Loaded the dataset and separated the features and labels.

2. **Data Splitting:**
   - Split the dataset into training and testing sets with an 80-20 split.

3. **Model Building:**
   - Defined a Sequential model with Keras, consisting of two layers: a fully connected layer and an output layer for binary classification.

4. **Model Compilation and Training:**
   - Compiled the model using the binary crossentropy loss function and Adam optimizer.
   - Trained the model over 100 epochs with a batch size of 50.

5. **Model Evaluation:**
   - Evaluated the model on the test set to determine its accuracy.

6. **Predictions and Confusion Matrix:**
   - Generated predictions on the test set and visualized the performance using a confusion matrix.

### Results
- **Accuracy:** The neural network model achieved an accuracy of 85.00%.
- **Confusion Matrix:** Visualized the number of true positive, true negative, false positive, and false negative predictions.

### Comparison with Other Models
- **Logistic Regression:** Achieved an accuracy of 84.62%.
- **MLPClassifier:** Achieved an accuracy of 86.54%.

### Files
- Implementation Code: [Q4.py](Code/Q4.py)
- Output Confusion Matrix: [output-Q4.png](Code/output-Q4.png)
- Detailed Report: [ML-Assignment4.pdf](ML-Assignment4.pdf)

## Conclusion
These assignments demonstrate the process of building and evaluating different classifiers for classifying spine conditions. The neural network's performance was comparable to logistic regression and slightly lower than sklearn's MLPClassifier. Further improvements could involve hyperparameter tuning, trying different network architectures, or using more advanced techniques like convolutional neural networks (CNNs) for enhanced performance.
