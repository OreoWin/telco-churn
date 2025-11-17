# Telco Customer Churn Analysis  
**Rebecca Li | UCLA Department of Statistics & Data Science**  

# 1. Introduction

Customer churn — the loss of existing subscribers — represents a major revenue risk for telecommunications companies. The goal of this project is to understand the behavioral and service-related factors driving churn, and provide modeling-ready insights for future predictive analysis (Logistic Regression & XGBoost & SVM). By examining distributions, feature relationships, and churn patterns, we aim to identify high-risk customer groups and actionable levers for churn reduction.

This project analyzes churn patterns in a Telco subscription dataset using:

- **R** for Exploratory Data Analysis (EDA)  
- **Python** for modelling Logistic Regression, XGBoost, and SVM  
- **Goal:** Identify key churn drivers, build predictive models, and provide actionable business insights.

---

# 2. Data Overview

The
 [TELCO Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)

 contains 7,043 customer records and 21 variables describing customer demographics, service subscriptions, contractual attributes, monthly billing, and a binary churn label. The dataset includes both numerical features (e.g., tenure, MonthlyCharges, TotalCharges) and multiple categorical variables related to phone, internet, and streaming services. 

Target variable: **Churn (Yes/No)**.

---

# 3. Exploratory Data Analysis (R)

EDA was conducted in **R (tidyverse + ggplot2)** to understand customer behavior and identify early indicators of churn.  
Key findings highlight imbalances in churn rates, differences in distributions of numeric variables, and a clear separation between churn and non-churn groups.



---

## 3.1 Churn Distribution

The dataset is imbalanced:  
- **73.4%** customers did **not** churn  
- **26.6%** customers **did** churn  

This imbalance is common in churn problems and motivates the use of AUC, recall, and precision instead of relying solely on accuracy.

### **Plot: Churn Distribution**

![Churn Distribution](report/figures/EDA_Churn_distribution_colored.png)

—
##3.2 Distribution of Numeric Variables
We examine three key numeric variables:
MonthlyCharges


tenure


TotalCharges


Observations:
Tenure has a reverse-J shape, typical in telecom retention patterns.


TotalCharges is positively skewed, consistent with long-tenure customers accumulating more charges.


MonthlyCharges has a fairly uniform spread.
![Numeric Variable Distribution](report/figures/EDA_Numerical_distribution_colored.png)


---

# 4. Logistic Regression (Baseline Model)

Logistic Regression is used as the baseline because it offers strong interpretability and sets a reference point for more complex models such as XGBoost and SVM.

### **Model Workflow**

- Standardized numerical variables  
- Trained on train split  
- Evaluated on test split using accuracy, AUC, precision, recall, and F1  


```python
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",    
    penalty="l2",
    random_state=42
)


log_reg.fit(X_train_scaled, y_train)


y_pred = log_reg.predict(X_test)
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]


print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))
print(classification_report(y_test, y_pred))



```
```text
ROC AUC: 0.4126848750588857

precision    recall  f1-score   support

           0       0.63      0.02      0.04      1033
           1       0.26      0.97      0.41       374

    accuracy                           0.27      1407
      1407
weighted avg       0.53      0.27      0.14      1407
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

## 5.2 XGBoost Hyperparameter Tuning (GridSearchCV)

Tuning aims to balance bias–variance tradeoff and improve generalization.

## 5.3 Model Evaluation 

## 5.4 Model Interpretation 

---

# 6. Support Vector Machine (SVM) Modelling 

SVM is a margin-based classifier that uses kernel transformations to identify nonlinear patterns in the churn data.  
Because SVM is highly sensitive to magnitudes of features, **standardization is required** to ensure balanced treatment across variables.

We evaluate:
1. **Baseline SVM**
2. **Tuned SVM (C, gamma GridSearchCV)**

---

## 6.1 Baseline SVM

SVM commonly offers several kernel choices:
1. Linear Kernel
Uses a straight-line (or hyperplane) boundary.
It works well when the data is approximately linearly separable or when the feature space is already high-dimensional after one-hot encoding. However, it cannot capture nonlinear churn patterns created by interactions between customer attributes.
2. Polynomial Kernel
Allows curved boundaries of polynomial form. Although more flexible than the linear kernel, it often introduces unnecessary complexity and tends to overfit medium-sized datasets such as churn data.
3. RBF (Radial Basis Function) Kernel
The RBF kernel maps points into a much higher-dimensional space where complex relationships become linearly separable. It produces smooth, nonlinear boundaries that adapt well to real-world structures in the data. RBF is generally the default and most robust choice for classification tasks involving heterogeneous or nonlinear feature interactions.

From EDA, we observed that customer churn patterns are rarely linearly separable. After one-hot encoding, the feature space becomes high-dimensional, and churn behavior depends on nonlinear interactions such as tenure × contract type × tech support status. The RBF kernel is well-suited to this scenario because it:
- captures nonlinear relationships between features
- handles complex customer behavior patterns
- avoids overfitting better than polynomial kernels
- consistently performs well in practical churn prediction tasks
For these reasons, we select the RBF kernel as the primary kernel for our SVM model.

## 6.2 Hyperparameter Tuning for SVM (GridSearchCV)

## 6.3 Model Evaluation 

## 6.4 Model Interpretation 

---

## 7. Model Comparison 
![Model Comparison](report/figures/model_comparison.png)







