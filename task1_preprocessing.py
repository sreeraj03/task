"""
Task 1: Data Understanding & Preprocessing
Objective: Check data handling skills
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TASK 1: DATA UNDERSTANDING & PREPROCESSING")
print("=" * 80)

# Load dataset
df = pd.read_csv('customer_churn_dataset.csv')

print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Dataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset Info:")
df.info()
print(f"\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\n2. MISSING VALUES ANALYSIS")
print("-" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_df)

# Handle missing values
print("\n3. HANDLING MISSING VALUES")
print("-" * 80)

# Fill numerical columns with median
for col in ['Age', 'Monthly_Charges']:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"✓ Filled {col} with median: {median_val:.2f}")

# Fill categorical columns with mode
for col in ['Gender', 'Contract_Type', 'Internet_Service']:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"✓ Filled {col} with mode: {mode_val}")

print(f"\n✓ Missing values after handling: {df.isnull().sum().sum()}")

# Data cleaning
print("\n4. DATA CLEANING")
print("-" * 80)
print(f"Duplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# Remove CustomerID (not a feature)
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)
    print("✓ Removed CustomerID column")

# Check data types
print("\nData types:")
print(df.dtypes)

# Exploratory Data Analysis
print("\n5. EXPLORATORY DATA ANALYSIS")
print("-" * 80)

# Churn distribution
churn_counts = df['Churn'].value_counts()
churn_rate = (churn_counts['Yes'] / len(df)) * 100
print(f"\nChurn Distribution:")
print(churn_counts)
print(f"Churn Rate: {churn_rate:.2f}%")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Churn distribution
axes[0, 0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Churn Distribution')

# Age distribution
axes[0, 1].hist(df['Age'], bins=20, edgecolor='black', color='skyblue')
axes[0, 1].set_title('Age Distribution')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Frequency')

# Monthly Charges distribution
axes[1, 0].hist(df['Monthly_Charges'], bins=20, edgecolor='black', color='lightgreen')
axes[1, 0].set_title('Monthly Charges Distribution')
axes[1, 0].set_xlabel('Monthly Charges')
axes[1, 0].set_ylabel('Frequency')

# Tenure distribution
axes[1, 1].hist(df['Tenure_Months'], bins=20, edgecolor='black', color='coral')
axes[1, 1].set_title('Tenure Distribution')
axes[1, 1].set_xlabel('Tenure (Months)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_visualization.png")
plt.close()

# Categorical features analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Categorical Features vs Churn', fontsize=16, fontweight='bold')

cat_features = ['Gender', 'Contract_Type', 'Internet_Service', 'Payment_Method']
for idx, col in enumerate(cat_features):
    row = idx // 3
    col_idx = idx % 3
    pd.crosstab(df[col], df['Churn']).plot(kind='bar', ax=axes[row, col_idx], rot=45)
    axes[row, col_idx].set_title(f'{col} vs Churn')
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('Count')

# Support Tickets
axes[1, 1].hist([df[df['Churn']=='No']['Support_Tickets'], 
                 df[df['Churn']=='Yes']['Support_Tickets']], 
                label=['No Churn', 'Churn'], bins=6, edgecolor='black')
axes[1, 1].set_title('Support Tickets vs Churn')
axes[1, 1].set_xlabel('Support Tickets')
axes[1, 1].legend()

axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: categorical_analysis.png")
plt.close()

# Encode categorical variables
print("\n6. ENCODING CATEGORICAL VARIABLES")
print("-" * 80)

label_encoders = {}
categorical_cols = ['Gender', 'Contract_Type', 'Internet_Service', 'Payment_Method', 'Churn']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Correlation heatmap
print("\n7. CORRELATION ANALYSIS")
print("-" * 80)
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_heatmap.png")
plt.close()

# Feature scaling
print("\n8. FEATURE SCALING")
print("-" * 80)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Tenure_Months', 'Monthly_Charges', 'Support_Tickets']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("✓ Scaled features:")
print(X.head())

# Save processed data
print("\n9. SAVING PROCESSED DATA")
print("-" * 80)
df.to_csv('processed_data.csv', index=False)
print("✓ Saved: processed_data.csv")

X.to_csv('X_features.csv', index=False)
print("✓ Saved: X_features.csv")

y.to_csv('y_target.csv', index=False)
print("✓ Saved: y_target.csv")

# Save encoders and scaler for deployment
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ Saved: label_encoders.pkl")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler.pkl")

print("\n" + "=" * 80)
print("✅ TASK 1 COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nSummary:")
print(f"  • Dataset: {len(df)} rows, {len(X.columns)} features")
print(f"  • Missing values: Handled")
print(f"  • Categorical encoding: Done")
print(f"  • Feature scaling: Done")
print(f"  • Visualizations: 3 files created")
print(f"  • Data ready for modeling!")
