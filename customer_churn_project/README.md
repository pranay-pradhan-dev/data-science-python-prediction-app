# Customer Churn Prediction

## Overview

This project implements an end-to-end Machine Learning solution for predicting telecom customer churn.

The solution covers the complete workflow from data understanding and exploratory analysis through model training, evaluation, interpretation, persistence, and deployment as a REST API.

The final solution includes:

- Data understanding and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Leakage-safe preprocessing
- Decision Tree classification
- Model comparison and overfitting analysis
- Feature importance and model interpretation
- Model persistence using Joblib
- REST API using FastAPI
- Input validation using Pydantic

---

## GitHub Repository

The complete source code for this project is available on GitHub:

[https://github.com/pranay-pradhan-dev/data-science-python-prediction-app](https://github.com/pranay-pradhan-dev/data-science-python-prediction-app)\

---

## Project Structure

```text
customer_churn_project/
│
├── data/
│   └── TelcoCustomerChurn.csv
│
├── notebook/
│   └── churn_analysis.ipynb
│
├── model/
│   └── churn_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── sample_request.json
```

> The local virtual environment (`venv/`) is not required as part of the project submission.

---

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset includes customer information such as:

- Gender
- Senior citizen status
- Partner and dependent status
- Tenure
- Phone service
- Internet service
- Online services
- Technical support
- Streaming services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn status

The target variable is:

```text
Churn
```

The target indicates whether a customer has churned.

The `customerID` field was excluded from the machine learning features because it is a customer identifier rather than a behavioral predictor.

---

## Data Cleaning

Initial data-quality checks included:

- Dataset shape and data types
- Missing values
- Duplicate records
- Unique values
- Target distribution
- Numerical and categorical statistics

### TotalCharges

`TotalCharges` was initially represented as a string column.

After attempting numeric conversion, 11 records contained invalid or missing `TotalCharges` values. Investigation showed that all 11 customers had a tenure of 0 months.

Because these customers had not yet accumulated historical charges, their missing `TotalCharges` values were assigned a value of 0 rather than removing the customer records or imputing an overall mean or median.

---

## Exploratory Data Analysis

EDA was performed to investigate customer churn patterns.

The analysis included:

1. Overall churn distribution
2. Churn by contract type
3. Tenure by churn status
4. Monthly charges by churn status
5. Churn by internet service
6. Churn by payment method
7. Churn by technical support

The visualizations were used to identify associations between customer characteristics and churn behavior.

EDA findings were treated as associations and not as evidence of causation.

---

## Feature Engineering

Two additional features were created.

### TotalServices

`TotalServices` represents the number of selected additional services subscribed to by a customer.

The following services were included:

- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

This provides an aggregated measure of customer service engagement.

### TenureGroup

Customers were grouped according to tenure:

```text
0-12 months
13-24 months
25-48 months
49-60 months
61+ months
```

This feature helps represent potentially different behavior between newer and long-standing customers.

---

## Train-Test Split

The data was separated into training and testing sets using:

```python
test_size=0.30
random_state=42
stratify=y
```

This resulted in:

- 70% training data
- 30% testing data

Stratification was used to maintain a similar churn-class distribution between the training and testing datasets.

---

## Data Preprocessing

A Scikit-learn `ColumnTransformer` and `Pipeline` were used for preprocessing.

The preprocessing workflow handles numerical and categorical features separately.

### Numerical Features

Missing numerical values are handled using median imputation.

### Categorical Features

Categorical features are processed using:

- Most-frequent-value imputation
- One-hot encoding

`OneHotEncoder` was configured with:

```python
handle_unknown="ignore"
```

This allows the preprocessing pipeline to handle previously unseen categories during prediction without immediately failing.

Preprocessing was fitted as part of the Machine Learning pipeline to help avoid data leakage.

---

## Model Training

Two Decision Tree configurations were trained and evaluated.

### Model 1: Baseline Decision Tree

The first model was an unrestricted Decision Tree:

```python
DecisionTreeClassifier(
    random_state=42
)
```

### Model 2: Controlled Decision Tree

The second model restricted tree complexity:

```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
```

The purpose of the controlled configuration was to reduce overfitting and improve generalization to unseen customer data.

---

## Model Evaluation

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Test-Set Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Decision Tree 1 - Baseline | 0.733 | 0.497 | 0.496 | 0.496 |
| Decision Tree 2 - Controlled | **0.794** | **0.615** | **0.602** | **0.608** |

The controlled Decision Tree outperformed the baseline model across all evaluated test metrics.

Recall was given particular consideration because a false negative represents a customer who actually churns but is predicted as a non-churner. Missing such customers may reduce the opportunity for potential retention
