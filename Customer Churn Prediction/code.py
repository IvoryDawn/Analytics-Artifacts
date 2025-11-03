import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('Customer Churn Predictor\Customer-Churn-Records.csv')

# Explore the dataset
# Dimension
print("Dimension: ", data.shape)
# Data Type of each column
print("Data Type of each column: ", data.dtypes)
# Missing value count
print("Missing value count: ", data.isnull().sum())
# Basic statistics of numerical features
print("Basic statistics: ", data.describe())
# Distribution of the target variable (Exited)
print("\nTarget Variable Distribution (Exited):")
print(data['Exited'].value_counts())

# Percentage distribution
print("\nPercentage Distribution:")
print(data['Exited'].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
data['Exited'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Customer Churn Distribution')
plt.xlabel('Exited (0 = Stayed, 1 = Left)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
data[numerical_cols].hist(bins=30, figsize=(20, 10))
plt.suptitle('Distribution of Numerical Features')
plt.show()

# Data Cleaning
# Columns that add NO predictive value (as per dataset description)
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
print("Columns to drop with reasons:")
print("- RowNumber: Just an index, no business value")
print("- CustomerId: Random identifier, no predictive power") 
print("- Surname: Name doesn't affect churn decision")

# Create a clean dataset by dropping these columns
data_clean = data.drop(columns=columns_to_drop)
print(f"\nOriginal shape: {data.shape}")
print(f"After cleaning: {data_clean.shape}")
print(f"\nRemaining columns: {list(data_clean.columns)}")

# EDA
# Complains
print("Churn rate by Complains: ")
print(data_clean.groupby('Complain')['Exited'].mean() * 100)
sns.barplot(x=data_clean['Complain'], y = data_clean['Exited'])
plt.show()

# Number of Products
prod_bins = [0, 1, 2, 4]
prod_labels = ['1 product', '2 products', '3+ products']
data_clean['Products_Tier'] = pd.cut(data_clean['NumOfProducts'], bins = prod_bins, labels = prod_labels, include_lowest=True, right=True)
sns.barplot(x=data_clean['Products_Tier'], y=data_clean['Exited'])
plt.show()

# Balance
data_clean['Balance_Tier'] = 'Zero Balance'
non_zero_balances = data_clean[data_clean['Balance'] > 0]['Balance']
quantiles = non_zero_balances.quantile([0.33, 0.66]).tolist()
low_bound = quantiles[0]
mid_bound = quantiles[1]
balance_bins = [0.0001, low_bound, mid_bound, data_clean['Balance'].max()]
balance_labels = ['Low Balance', 'Medium Balance', 'High Balance']
data_clean.loc[data_clean['Balance'] > 0, 'Balance_Tier'] = pd.cut(data_clean.loc[data_clean['Balance'] > 0, 'Balance'], bins=balance_bins, labels=balance_labels, include_lowest=True, right=True)
sns.barplot(x= data_clean['Balance_Tier'], y = data_clean['Exited'])
plt.show()

# Age
age_bins = [0, 26, 46, 61, 100] 
age_labels = ['Young Adult', 'Prime Working Age', 'Pre-Retirement', 'Senior/Retiree']
data_clean['Age_Tier'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)
print("Churn Rate by Age: ")
print(data_clean.groupby('Age_Tier')['Exited'].mean() * 100)
sns.barplot(x = data_clean['Age_Tier'], y = data_clean['Exited'])
plt.show()

# Geography
print("Churn Rate by Geography:")
print(data_clean.groupby('Geography')['Exited'].mean() * 100)
sns.barplot(x = data_clean['Geography'], y = data_clean['Exited'])
plt.show()

# Gender
print("Churn Rate by Gender: ")
print(data_clean.groupby('Gender')['Exited'].mean() * 100)
sns.barplot(x = data_clean['Gender'], y = data_clean['Exited'])
plt.show()

from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Numerical variables scaling
num_cols = ['CreditScore', 'EstimatedSalary', 'Tenure', 'Point Earned', 'Balance', 'Satisfaction Score']
scaler = StandardScaler()
data_clean[num_cols] = scaler.fit_transform(data_clean[num_cols])
data_clean = data_clean.drop(columns=['Age', 'NumOfProducts', 'Balance', 'Card Type', 'Satisfaction Score'])
# Categorical varibles encoding
data_clean['Gender_Male'] = data_clean['Gender'].map({'Male': 1, 'Female': 0})
data_clean = data_clean.drop('Gender', axis=1)
cat_cols = ['Age_Tier', 'Geography', 'Balance_Tier', 'Products_Tier']
data_final = pd.get_dummies(data_clean, columns=cat_cols,drop_first=True)
print("\nCategorical Features Encoded Successfully.")
print(data_final.columns.to_list())

# Defining features(x) and target(y)
y = data_final['Exited']
leakage_cols = ['Exited', 'Complain', 'Products_Tier_3+ products']
x = data_final.drop(columns=leakage_cols, axis=1)
print("Features (X) shape:", x.shape)
print("Target (y) shape:", y.shape)

# Splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

# Training the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 200, random_state = 42, class_weight = 'balanced', max_depth = 10)
print("Training the Random Forest CLassifier: ")
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
y_predict = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]
print("Model Evaluation: ")
print("Accuracy: ", accuracy_score(y_test, y_predict))
print("AUC-ROC score: ", roc_auc_score(y_test, y_prob))
print("Classification Report: \n", classification_report(y_test, y_predict))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_predict))

# Feature Importance
importance = model.feature_importances_
feature_names = x.columns
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title('Top 10 Feature Importance (Gini Importance)')
plt.show()

# Business Impact Analysis
assumed_annual_profit = 500 # $500 per customer 
total_customers = len(y_test)
churn_rate = y_test.mean()
predicted_churn = y_predict.sum()
retention_rate = 1 - churn_rate
if churn_rate > 0 :
    avg_cust_lifespan = 1 / churn_rate
else :
    avg_cust_lifespan = float(np.inf)

print("BUSINESS IMAPCT ANALYSIS: ")
print("Total number of customers in the set: ", total_customers)
print("Actual churn rate : ", churn_rate)
print("Predicted churn : ", predicted_churn, " customers")
print("Retention rate : ", retention_rate)
print("Average customer lifespan : ", avg_cust_lifespan)

# Model comparison
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

models = {'Logistic Regression' : LogisticRegression(random_state = 42, class_weight = 'balanced'), 'Gradient Boosting Classifier' : GradientBoostingClassifier(random_state = 42), 'Random Forest' : RandomForestClassifier(n_estimators = 200, random_state = 42, class_weight = 'balanced', max_depth = 10)}
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    results[name] = {'accuracy' : accuracy_score(y_test, y_predict), 'auc_roc' : roc_auc_score(y_test, y_prob)}

results_df = pd.DataFrame(results).T
print("\n=== MODEL COMPARISON ===")
print(results_df.sort_values('auc_roc', ascending=False))

# Define risk tiers
risk_tier = pd.cut(x=y_prob,bins=[0, 0.3, 0.7, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk'])
print("\n=== CUSTOMER RISK SEGMENTS ===")
print(risk_tier.value_counts())
high_risk_customers = x_test[y_prob > 0.7]  # High-risk customer profile
print(f"\nHigh-risk customers to target: {len(high_risk_customers)}")

print("=== PROJECT EXECUTIVE SUMMARY ===")
print("🎯 BUSINESS PROBLEM: 20.4% customer churn costing millions")
print("🔧 SOLUTION: ML model predicting churn with 82.2% AUC")
print("📈 KEY FINDINGS:")
print("   - Germany has 2x higher churn than other countries")
print("   - Complaint handling is critical (99.5% churn rate)")
print("   - Pre-retirement age group most vulnerable")
print("💡 RECOMMENDATIONS:")
print("   - Target 175 high-risk customers immediately")
print("   - Regional retention campaigns for Germany")
print("   - Product development for 46-60 age group")

# Save best model
import joblib

best_model = GradientBoostingClassifier(random_state=42)
best_model.fit(x_train, y_train)

joblib.dump(best_model, 'churn_predictor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved for production use!")
