
# Machine Learning Assignment 2
## Employee Attrition Classification

---

## 1. Problem Statement

Employee attrition is an important problem for organizations because unexpected employee turnover can increase recruitment costs, reduce productivity, and affect workforce planning.

The objective of this project is to develop and compare multiple machine learning classification models for predicting whether an employee is likely to leave the organization.

The target variable is **Attrition**, which has two classes:

- No — Employee does not leave
- Yes — Employee leaves

Five machine learning classification algorithms were implemented and evaluated using multiple performance metrics.

---

## 2. Dataset Description

The dataset used for this project is the IBM HR Analytics Employee Attrition & Performance dataset obtained from Kaggle.

The dataset contains information about employees, including demographic information, job characteristics, compensation, job satisfaction, work experience, and other employment-related attributes.

### Dataset Statistics

- Number of instances: 1470
- Number of original features/columns: 35
- Problem type: Binary Classification
- Target variable: Attrition

The target variable was converted into numerical form:

- No → 0
- Yes → 1

### Data Preprocessing

The following preprocessing steps were performed:

1. Checked for missing values.
2. Checked for duplicate records.
3. Removed irrelevant identifier/constant columns:
   - EmployeeNumber
   - EmployeeCount
   - Over18
   - StandardHours
4. Separated the target variable from the input features.
5. Numerical features were standardized using StandardScaler.
6. Categorical features were converted using OneHotEncoder.
7. The dataset was divided into training and testing sets using an 80:20 split.
8. Stratification was used to preserve the target class distribution.

---

## 3. Machine Learning Models

The following five classification algorithms were implemented:

### 1. Logistic Regression

Logistic Regression was used as a linear classification baseline for predicting employee attrition.

### 2. Decision Tree

Decision Tree is a rule-based tree model capable of learning nonlinear relationships between features and the target.

### 3. K-Nearest Neighbors (KNN)

KNN classifies observations based on the classes of nearby training observations.

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem and the assumption of conditional independence between features.

### 5. Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to improve predictive performance and robustness.

---

## 4. Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

These metrics provide a broader evaluation than accuracy alone, especially because employee attrition is an imbalanced classification problem.

---

## 5. Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8605 | 0.8115 | 0.6154 | 0.3404 | 0.4384 | 0.3871 |
| Decision Tree | 0.7653 | 0.6105 | 0.3103 | 0.3830 | 0.3429 | 0.2036 |
| KNN | 0.8435 | 0.5946 | 0.5385 | 0.1489 | 0.2333 | 0.2222 |
| Naive Bayes | 0.6463 | 0.7032 | 0.2605 | 0.6596 | 0.3735 | 0.2265 |
| Random Forest | 0.8503 | 0.8037 | 0.6364 | 0.1489 | 0.2414 | 0.2563 |

---

## 6. Observations

### Logistic Regression

Logistic Regression achieved the highest accuracy (0.8605), AUC (0.8115), F1-score (0.4384), and MCC (0.3871). It provided the most balanced overall performance among the evaluated models.

### Decision Tree

Decision Tree achieved an accuracy of 0.7653 and an AUC of 0.6105. Its recall was 0.3830, but its precision, F1-score, and MCC were relatively low compared with Logistic Regression.

### KNN

KNN achieved relatively high accuracy (0.8435), but its AUC was only 0.5946 and its recall was 0.1489. Therefore, although it classified many observations correctly overall, it detected relatively few actual attrition cases.

### Naive Bayes

Naive Bayes achieved the highest recall of 0.6596, meaning it identified the largest proportion of actual attrition cases. However, it had low precision (0.2605) and the lowest accuracy (0.6463), indicating a relatively high number of false positive predictions.

### Random Forest

Random Forest achieved the highest precision of 0.6364 and a strong AUC of 0.8037. However, its recall was only 0.1489, resulting in a relatively low F1-score of 0.2414.

---

## 7. Overall Winner

### Logistic Regression

Logistic Regression was selected as the overall winner.

It achieved:

- Highest Accuracy: 0.8605
- Highest AUC: 0.8115
- Highest F1 Score: 0.4384
- Highest MCC: 0.3871

Although Random Forest achieved slightly higher precision and Naive Bayes achieved the highest recall, Logistic Regression provided the most balanced overall performance across the evaluation metrics.

---

## 8. Streamlit Application

A Streamlit application was developed to provide an interactive interface for evaluating the trained machine learning models.

The application provides:

1. CSV file upload
2. Machine learning model selection
3. Accuracy
4. AUC
5. Precision
6. Recall
7. F1 Score
8. MCC
9. Confusion matrix
10. Classification report
11. Prediction results
12. Downloadable prediction results

---

## 9. Repository Structure

```text
ML-Assignment-2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
