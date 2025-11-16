# Telco Customer Churn Analysis  
**Rebecca Li | UCLA Department of Statistics & Data Science**  

# 1. Introduction

Customer churn — the loss of existing subscribers — represents a major revenue risk for telecommunications companies. The goal of this exploratory analysis is to understand the behavioral and service-related factors driving churn, and provide modeling-ready insights for future predictive analysis (XGBoost & SVM). By examining distributions, feature relationships, and churn patterns, we aim to identify high-risk customer groups and actionable levers for churn reduction.

This project analyzes churn patterns in a Telco subscription dataset using:

- **R** for Exploratory Data Analysis (EDA)  
- **Python** for modelling Logistic Regression, XGBoost, and SVM  
- **Goal:** Identify key churn drivers, build predictive models, and provide actionable business insights.

---

# 2. Data Overview

The Telco Customer Churn dataset contains 7,043 customer records and 21 variables describing customer demographics, service subscriptions, contractual attributes, monthly billing, and a binary churn label. The dataset includes both numerical features (e.g., tenure, MonthlyCharges, TotalCharges) and multiple categorical variables related to phone, internet, and streaming services. 

Target variable: **Churn (Yes/No)**.

---

# 3. Exploratory Data Analysis (R)

EDA was conducted using `tidyverse` and `ggplot2`.  
Key findings:

- Customers with **Month-to-Month** contracts churn at the highest rate.  
- **Short-tenure** customers are significantly more likely to churn.  
- Customers using **electronic check** payment show higher churn risk.  
- Senior citizens have a slightly higher churn probability.  



```r
ggplot(df, aes(tenure, fill = Churn)) +
    geom_histogram(position = "dodge", bins = 30)
```

---

# 4. Logistic Regression (Baseline Model)

Logistic Regression is used as the baseline because it offers strong interpretability and sets a reference point for more complex models such as XGBoost and SVM.

### **Model Workflow**

- Standardized numerical variables  
- Trained on train split  
- Evaluated on test split using accuracy, AUC, precision, recall, and F1  

### **Python Code (Baseline)**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
y_prob = logreg.predict_proba(X_test_scaled)[:, 1]

print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))
```

---

# 5. XGBoost Modelling

XGBoost is a tree-based gradient boosting algorithm known for its strong performance on tabular datasets.  
In churn prediction tasks, it often outperforms linear models by capturing nonlinear interactions between features such as tenure, contract type, and billing behavior.

We evaluate two variants:
1. **Baseline XGBoost**
2. **Tuned XGBoost (GridSearchCV)**

---

## 5.1 Baseline XGBoost

The baseline model uses default hyperparameters.  
This provides a reference to evaluate how much tuning improves performance.

### **Python Code (Baseline)**

```python
import xgboost as xgb
from sklearn.metrics import roc_auc_score

xgb_model = xgb.XGBClassifier(
    eval_metric="logloss", 
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)

xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
print("XGBoost Baseline AUC:", roc_auc_score(y_test, xgb_prob))


---

# 6. Support Vector Machine (SVM)

SVM is a margin-based classifier that uses kernel transformations to identify nonlinear patterns in the churn data.  
Because SVM is highly sensitive to magnitudes of features, **standardization is required** to ensure balanced treatment across variables.

We evaluate:
1. **Baseline SVM**
2. **Tuned SVM (C, gamma GridSearchCV)**

---

## 6.1 Baseline SVM

The baseline SVM uses an RBF kernel with default hyperparameters.  
This gives us a starting point to later assess how tuning improves performance.







