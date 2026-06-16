# Customer Churn Prediction using Artificial Neural Network (ANN)

## 1. Project Objective

The objective of this project is to predict customer churn for a telecom company using an Artificial Neural Network (ANN). Customer churn refers to customers leaving the company. Early identification of churn helps businesses take retention actions and reduce customer loss.

**Target Variable:**
- Churn = 1 → Customer leaves
- Churn = 0 → Customer stays

---

## 2. Dataset Information

**Dataset:** Telco Customer Churn Dataset

**Total Records:** 7043

**Features Used:**
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

**Target Variable:**
- Churn

---

## 3. Exploratory Data Analysis (EDA)

The following analyses were performed:

### Churn Distribution
- Dataset was imbalanced.
- Non-Churn customers were significantly higher than Churn customers.

### Tenure vs Churn
- Customers with lower tenure showed higher churn probability.
- Long-term customers were less likely to churn.

### Monthly Charges vs Churn
- Customers with higher monthly charges tended to churn more frequently.

### Contract Type Analysis
- Month-to-Month contract customers had the highest churn rate.
- One-Year and Two-Year contracts showed lower churn.


---

## 4. Data Preprocessing

### Customer ID Removal

CustomerID was removed because it does not contribute to churn prediction.

```python
df.drop('customerID', axis=1, inplace=True)
```

### Missing Value Handling

- Missing values in TotalCharges were identified and removed.
- Data types were corrected.

### Categorical Encoding

Categorical variables were converted using One-Hot Encoding.

```python
df = pd.get_dummies(df, drop_first=True)
```

---

## 5. Feature Engineering

### Feature Created: Number of Services

A new feature was created to represent how many telecom services a customer uses.

Examples:
- Phone Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

This feature helped summarize customer engagement.

### Feature Scaling

StandardScaler was used to standardize feature values.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 6. Train-Test Split

The dataset was split using Stratified Sampling.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Training Data
- 5634 Records

### Testing Data
- 1409 Records

---

## 7. Class Imbalance Problem

Original Distribution:

- Class 0 (No Churn): 5174
- Class 1 (Churn): 1869

The dataset was imbalanced.

---

## 8. SMOTE for Data Balancing

SMOTE was applied only on training data.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)
```

Balanced Training Data:

- Class 0 = 4139
- Class 1 = 4139

---

## 9. ANN Model Architecture

### Input Layer

- 33 Features

### Hidden Layer 1

- Dense(64)
- BatchNormalization
- ReLU Activation
- Dropout(0.3)

### Hidden Layer 2

- Dense(32)
- BatchNormalization
- ReLU Activation
- Dropout(0.3)

### Output Layer

- Dense(1)
- Sigmoid Activation

### Architecture Used

```python
model = Sequential()

model.add(Input(shape=(33,)))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))
```

---

## 10. Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)
```

---

## 11. Model Training

### Early Stopping

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

### Training Parameters

- Epochs = 100
- Batch Size = 32
- Validation Split = 20%
- Early Stopping Applied

---

## 12. Model Evaluation

### Test Results

| Metric | Value |
|----------|----------|
| Accuracy | 77.08% |
| Precision | 56.17% |
| Recall | 62.03% |
| Loss | 0.4425 |

---

## 13. Confusion Matrix

```
[[854 181]
 [142 232]]
```

### Interpretation

- True Negatives = 854
- False Positives = 181
- False Negatives = 142
- True Positives = 232

The model successfully identified 232 churn customers while missing 142 churn customers.

---

## 14. Classification Report

### Class 0 (No Churn)

| Metric | Score |
|----------|----------|
| Precision | 0.86 |
| Recall | 0.83 |
| F1-Score | 0.84 |

### Class 1 (Churn)

| Metric | Score |
|----------|----------|
| Precision | 0.56 |
| Recall | 0.62 |
| F1-Score | 0.59 |

### Overall Performance

| Metric | Score |
|----------|----------|
| Accuracy | 0.77 |
| Macro F1 Score | 0.72 |
| Weighted F1 Score | 0.77 |

---

## 15. Key Findings

### Customers likely to churn:

- Customers with low tenure
- Customers with Month-to-Month contracts
- Customers with high monthly charges
- Customers using fewer value-added services

### Customers less likely to churn:

- Long-term customers
- Customers with annual contracts
- Customers using multiple telecom services

---

## 16. Improvements Made During Project

1. Removed CustomerID.
2. Handled missing values.
3. Performed One-Hot Encoding.
4. Conducted EDA and churn analysis.
5. Created feature engineering variables.
6. Applied Stratified Train-Test Split.
7. Handled class imbalance using SMOTE.
8. Applied StandardScaler.
9. Added Batch Normalization.
10. Added Dropout layers.
11. Applied Early Stopping.
12. Evaluated using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

---

## 17. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow
- Keras
- Imbalanced-Learn (SMOTE)
- Jupyter Notebook

---

## 18. Conclusion

An Artificial Neural Network (ANN) was developed to predict telecom customer churn.

Several improvements were incorporated, including:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- SMOTE for class balancing
- StandardScaler for feature scaling
- Batch Normalization
- Dropout Regularization
- Early Stopping

The final model achieved:

- Accuracy: 77.08%
- Precision: 56.17%
- Recall: 62.03%
- F1-Score: 59%

The model successfully identified a majority of churn customers and can be used as a customer retention support system. This project demonstrates practical skills in data preprocessing, feature engineering, deep learning, ANN development, model evaluation, and business problem solving.
