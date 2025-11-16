# Telco Customer Churn Analysis  
**Exploratory Data Analysis (R) + Machine Learning Modelling (Python)**  

Rebecca Li | UCLA MASDS
---

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

### Example R Code

```r
ggplot(df, aes(tenure, fill = Churn)) +
    geom_histogram(position = "dodge", bins = 30)

