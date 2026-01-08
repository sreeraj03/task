"""
Task 2: Model Building (Core AI Skill)
Option A - Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TASK 2: MODEL BUILDING - MACHINE LEARNING")
print("=" * 80)

# Load processed data
X = pd.read_csv('X_features.csv')
y = pd.read_csv('y_target.csv').values.ravel()

print(f"\nDataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Target distribution: {np.bincount(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Store models and results
models = {}
results = []

print("\n" + "=" * 80)
print("TRAINING AND EVALUATING MODELS")
print("=" * 80)

# 1. Logistic Regression
print("\n1. LOGISTIC REGRESSION")
print("-" * 80)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]

lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, zero_division=0)
lr_rec = recall_score(y_test, lr_pred, zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)
lr_auc = roc_auc_score(y_test, lr_proba)

print(f"Accuracy:  {lr_acc:.4f}")
print(f"Precision: {lr_prec:.4f}")
print(f"Recall:    {lr_rec:.4f}")
print(f"F1-Score:  {lr_f1:.4f}")
print(f"ROC-AUC:   {lr_auc:.4f}")

models['Logistic Regression'] = lr_model
results.append({
    'Model': 'Logistic Regression',
    'Accuracy': lr_acc,
    'Precision': lr_prec,
    'Recall': lr_rec,
    'F1-Score': lr_f1,
    'ROC-AUC': lr_auc
})

# 2. Random Forest
print("\n2. RANDOM FOREST")
print("-" * 80)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, zero_division=0)
rf_rec = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"Accuracy:  {rf_acc:.4f}")
print(f"Precision: {rf_prec:.4f}")
print(f"Recall:    {rf_rec:.4f}")
print(f"F1-Score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_auc:.4f}")

models['Random Forest'] = rf_model
results.append({
    'Model': 'Random Forest',
    'Accuracy': rf_acc,
    'Precision': rf_prec,
    'Recall': rf_rec,
    'F1-Score': rf_f1,
    'ROC-AUC': rf_auc
})

# 3. XGBoost
print("\n3. XGBOOST")
print("-" * 80)
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
xgb_rec = recall_score(y_test, xgb_pred, zero_division=0)
xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
xgb_auc = roc_auc_score(y_test, xgb_proba)

print(f"Accuracy:  {xgb_acc:.4f}")
print(f"Precision: {xgb_prec:.4f}")
print(f"Recall:    {xgb_rec:.4f}")
print(f"F1-Score:  {xgb_f1:.4f}")
print(f"ROC-AUC:   {xgb_auc:.4f}")

models['XGBoost'] = xgb_model
results.append({
    'Model': 'XGBoost',
    'Accuracy': xgb_acc,
    'Precision': xgb_prec,
    'Recall': xgb_rec,
    'F1-Score': xgb_f1,
    'ROC-AUC': xgb_auc
})

# Model comparison
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find best model
best_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f})")

# Save results
results_df.to_csv('model_results.csv', index=False)
print("\n✓ Saved: model_results.csv")

# Visualizations
print("\nCreating visualizations...")

# 1. Model comparison bar chart
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    row = idx // 2
    col = idx % 2
    axes[row, col].bar(results_df['Model'], results_df[metric], color=['#3498db', '#2ecc71', '#e74c3c'])
    axes[row, col].set_title(metric, fontweight='bold')
    axes[row, col].set_ylabel(metric)
    axes[row, col].set_ylim(0, 1)
    axes[row, col].tick_params(axis='x', rotation=45)
    for i, v in enumerate(results_df[metric]):
        axes[row, col].text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")
plt.close()

# 2. Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

predictions = [lr_pred, rf_pred, xgb_pred]
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']

for idx, (pred, name) in enumerate(zip(predictions, model_names)):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices.png")
plt.close()

# 3. ROC Curves
plt.figure(figsize=(10, 8))
probas = [lr_proba, rf_proba, xgb_proba]
colors = ['#3498db', '#2ecc71', '#e74c3c']

for proba, name, color in zip(probas, model_names, colors):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curves.png")
plt.close()

# 4. Feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Feature Importance', fontsize=16, fontweight='bold')

# Random Forest
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], color='#2ecc71')
axes[0].set_title('Random Forest Feature Importance')
axes[0].set_xlabel('Importance')
axes[0].invert_yaxis()

# XGBoost
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'], color='#e74c3c')
axes[1].set_title('XGBoost Feature Importance')
axes[1].set_xlabel('Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.close()

# Save feature importance
rf_importance.to_csv('rf_feature_importance.csv', index=False)
xgb_importance.to_csv('xgb_feature_importance.csv', index=False)
print("✓ Saved: rf_feature_importance.csv")
print("✓ Saved: xgb_feature_importance.csv")

# Save best model
best_model = models[best_model_name]
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n✓ Saved best model ({best_model_name}): best_model.pkl")

# Save all models
with open('all_models.pkl', 'wb') as f:
    pickle.dump(models, f)
print("✓ Saved all models: all_models.pkl")

print("\n" + "=" * 80)
print("✅ TASK 2 COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f}")
print(f"   Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
print(f"   ROC-AUC:  {results_df.loc[best_idx, 'ROC-AUC']:.4f}")
