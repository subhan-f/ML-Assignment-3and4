# Machine Learning Assignment

## Introduction
This report presents the implementation and results of two tasks involving linear regression and logistic regression, respectively. The first task involves predicting salaries based on position levels and admission chances based on various metrics using linear regression. The second task involves classifying whether a human spine is normal or abnormal using logistic regression.

## Question 1: Linear Regression

### Implementation Details
#### Position Level vs. Salary Prediction using Normal Equations
- **Data Loading**: The `posal.csv` file is loaded to obtain position level and salary data.
- **Feature Engineering**: Polynomial features up to degree 4 are created and normalized.
- **Model Training**: Using normal equations, we calculate the theta coefficients.
- **Model Comparison**: The normal equations method is compared with the `LinearRegression` model from `sklearn`.

#### Admission Chances Prediction using Normal Equations
- **Data Loading**: The `admit.csv` file is loaded to obtain features related to GRE score, TOEFL score, etc.
- **Data Splitting**: The data is split into training and testing sets.
- **Model Training**: Normal equations are used to predict admission chances.
- **Model Comparison**: The results are compared with the `LinearRegression` model from `sklearn`.

### Results
- **Position Level vs Salary**:
  - **Prediction using Normal Equations**: `pred_sal_normal_eq`
  - **Prediction using Sklearn LinearRegression**: `pred_sal_sklearn`
  - **Normal Equations Coefficients**: `theta_normal_eq`
  - **Sklearn Coefficients**: `theta_sklearn`
- **Admission Chances**:
  - **R2 score using Normal Equations**: `r2_normal_eq`
  - **R2 score using Sklearn LinearRegression**: `r2_sklearn`
 ![Q1-Figure](https://github.com/subhan-f/ML-Assignment-2/assets/67074140/c2af9f72-c5bd-4475-98cd-41c5b10e70e0)
![Q1-Output](https://github.com/subhan-f/ML-Assignment-2/assets/67074140/04913341-d9ca-46a6-b285-cb42dee03803)



## Question 2: Logistic Regression

### Implementation Details
- **Data Loading**: The `spine.csv` file is loaded to obtain features and labels.
- **Data Splitting**: The data is split into training and testing sets.
- **Model Training**: A logistic regression model is implemented from scratch using gradient descent.
- **Model Comparison**: The scratch implementation is compared with the `LogisticRegression` model from `sklearn`.

### Results
### Results
- **Accuracy (Scratch)**: `accuracy_scratch`
- **Confusion Matrix (Scratch)**: `conf_matrix_scratch`

- **Accuracy (Built-in)**: `accuracy_builtin`
- **Confusion Matrix (Built-in)**: `conf_matrix_builtin`
  
![Q2-Output](https://github.com/subhan-f/ML-Assignment-2/assets/67074140/d674e38a-6372-4334-ac9c-a33da47a1901)


## Conclusion
In both linear and logistic regression tasks, the scratch implementations closely matched the performance of the sklearn models, demonstrating the correctness of our manual implementations. The linear regression model effectively predicted salaries and admission chances, while the logistic regression model accurately classified spine conditions. These implementations provided a deeper understanding of the underlying algorithms used in machine learning models.

## Comments
- **Linear Regression**:
- The normal equations method for predicting salaries and admission chances yielded accurate predictions comparable to sklearn's `LinearRegression`.
- R2 scores for both implementations indicated high accuracy.
- **Logistic Regression**:
- The scratch implementation for classifying spine conditions achieved similar accuracy to sklearn's `LogisticRegression`.
- Confusion matrices for both implementations showed comparable results, highlighting the effectiveness of our custom logistic regression model.

By understanding and implementing these algorithms from scratch, we gained insights into the mathematical foundations and practical applications of machine learning techniques.
