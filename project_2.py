import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix
from xgboost import plot_importance


df = pd.read_csv('customer_churn_dataset.csv')

print(df.head())
print(df.describe())

#data cleaning
print('null values: ',df.isnull().sum())

print('duplicate values : ',df.duplicated().sum())


#feature engineering

df['avg_monthly_uasge'] = df['total_usage'] / df['tenure_months']

print(df['avg_monthly_uasge'].head())

df['complaint_rate'] = df['support_tickets'] / df['tenure_months']

print(df['complaint_rate'].head())


data = pd.get_dummies(df, columns=['region', 'plan_type'])

print(data.head())

df['churn'] = df['churn'].replace({0 : 1 , 1: 0})

print(df['churn'].value_counts())

dx = df.to_csv('final_project2_dataset.csv',index=False)

df['plan_type'] = df['plan_type'].astype('category').cat.codes.astype(float)


#train model
X = df[['avg_monthly_uasge','plan_type','last_login_days','complaint_rate']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression()
xgb_model = XGBClassifier()

lr_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

print("Logistic Regression F1:", f1_score(y_test, y_pred_lr))
print("XGBoost F1:", f1_score(y_test, y_pred_xgb))

print("Confusion Matrix (XGBoost):")
print(confusion_matrix(y_test, y_pred_xgb))

from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(xgb_model)
plt.show()


#Logistic Regression
features = X.columns
coefficients = lr_model.coef_[0]

plt.figure(figsize=(8, 5))
plt.barh(features, coefficients)
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Feature Importance')
plt.tight_layout()
plt.show()