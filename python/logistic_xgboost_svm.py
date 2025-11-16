#setup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt

#load data
df = pd.read_csv('telco_cleaned.csv')
print(df.head())

X = df.drop(["Churn","churn_binary", "customerID"], axis=1)  
y = df["churn_binary"]  

# Encode categorical features 
categorical_cols = X.select_dtypes(include=["object"]).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"\nEncoded {len(categorical_cols)} categorical columns") #16
print(f"Total features after encoding: {X_encoded.shape[1]}") #30

#train/test split (BEFORE scaling to avoid data leakage!)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

#scaling for SVM and Logistic Regression (fit on train, transform test)
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

#logistic regression - baseline model
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

#Logistic Regression: AUC = 0.41, accuracy = 0.27
#data is non-linear so logistical regression failed.

#-----------------------------------------------
#XGBoost
#-----------------------------------------------

#baseline XGBoost model
from xgboost import XGBClassifier

# XGBoost does NOT use scaling — use raw encoded X_train/X_test
X_train_xgb = X_train  
X_test_xgb = X_test

# handle imbalance: ratio = (# non-churn) / (# churn)
neg, pos = y_train.value_counts()
scale_pos_weight = neg / pos
print("scale_pos_weight:", scale_pos_weight) #2.76


xgb_model = XGBClassifier(
    n_estimators=300, #number of trees
    learning_rate=0.05,#prevent overfitting
    max_depth=5, #depth of each tree
    subsample=0.8, #fraction of samples used for each tree
    colsample_bytree=0.8, #fraction of features used for each tree
    scale_pos_weight=scale_pos_weight,   # imbalance correction
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train_xgb, y_train)

y_pred_xgb = xgb_model.predict(X_test_xgb)
y_pred_prob_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]

print("XGBoost ROC AUC:", roc_auc_score(y_test, y_pred_prob_xgb))
print(classification_report(y_test, y_pred_xgb))

#baseline XGBoost model: AUC = 0.826, accuracy = 0.7421

#Hyperparameter Tuning (GridSearchCV)
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

#Tuned XGBoost

#GridSearchCV (optimize ROC AUC)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_xgb, y_train)

print("Best params:", grid_search.best_params_)
#{'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}

#max_depth=3 → shallower trees → less overfit
#learning_rate=0.05 → stable learning
#n_estimators=200 → enough complexity without noise
#colsample_bytree=0.8, subsample=0.8 → regularization

print("Best CV ROC AUC:", grid_search.best_score_)
#0.849

best_xgb = grid_search.best_estimator_
y_pred_best = best_xgb.predict(X_test_xgb)
y_pred_best_prob = best_xgb.predict_proba(X_test_xgb)[:, 1]

print("Test ROC AUC (best XGB):", roc_auc_score(y_test, y_pred_best_prob))#0.838
print(classification_report(y_test, y_pred_best))

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test_xgb)
#shap.summary_plot(shap_values, X_test_xgb, plot_type="dot")
#shap.summary_plot(shap_values, X_test_xgb, plot_type="bar")
#shap.dependence_plot("tenure", shap_values, X_test_xgb)

#Best XGBoost model: Gain, Cover, Weight (Frequency)
from xgboost import plot_importance

#plot_importance(best_xgb, importance_type='gain', max_num_features=20)
#plt.title("XGBoost Feature Importance — Gain")
#plt.show()

#plot_importance(best_xgb, importance_type='cover', max_num_features=20)
#plt.title("XGBoost Feature Importance — Cover")
#plt.show()

#plot_importance(best_xgb, importance_type='weight', max_num_features=20)
#plt.title("XGBoost Feature Importance — Weight (Frequency)")
#plt.show()

#-----------------------------------------------
#SVM
#-----------------------------------------------
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

#Baseline RBF Kernel SVM Model
X_train_svm = X_train_scaled
X_test_svm = X_test_scaled

svm_rbf = SVC(
    kernel="rbf",
    C=1.0,             # default
    gamma="scale",      # default RBF gamma
    probability=True,   # needed for AUC
    class_weight="balanced",  # important for imbalanced churn data
    random_state=42
)

svm_rbf.fit(X_train_svm, y_train)

y_pred_svm = svm_rbf.predict(X_test_svm)
y_pred_svm_prob = svm_rbf.predict_proba(X_test_svm)[:, 1]

print("SVM (RBF) ROC AUC:", roc_auc_score(y_test, y_pred_svm_prob)) #0.81
print(classification_report(y_test, y_pred_svm))

#Tuned SVM

#GridSearchCV (optimize ROC AUC)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid_svm = {
    "C": [0.1, 1, 10, 50],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf"],
    "class_weight": ["balanced"]
}

svm_grid = GridSearchCV(
    estimator=svm_rbf,
    param_grid=param_grid_svm,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train_svm, y_train)

print("Best SVM params:", svm_grid.best_params_)
print("Best CV ROC AUC:", svm_grid.best_score_) #0.843

best_svm = svm_grid.best_estimator_

y_pred_best_svm = best_svm.predict(X_test_svm)
y_pred_best_svm_prob = best_svm.predict_proba(X_test_svm)[:, 1]

print("Test ROC AUC (best SVM):", roc_auc_score(y_test, y_pred_best_svm_prob)) #0.833
print(classification_report(y_test, y_pred_best_svm))

#Evaluation on best SVM model 

#Support Vector Count (Model Complexity)
print("Support vectors per class:", best_svm.n_support_)
total_sv = best_svm.support_.shape[0]
print("Total support vectors:", total_sv)
percent_sv = total_sv / X_train_svm.shape[0]
print(f"Percentage of support vectors: {percent_sv:.2%}")

#Decision Function Distribution (Model Confidence)
decision_scores = best_svm.decision_function(X_test_svm)
plt.hist(decision_scores[y_test==0], bins=30, alpha=0.6, label="Non-churn")
plt.hist(decision_scores[y_test==1], bins=30, alpha=0.6, label="Churn")
plt.title("SVM Decision Function Distribution")
plt.xlabel("Decision Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()