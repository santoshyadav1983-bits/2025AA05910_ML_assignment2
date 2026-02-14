"""
Classification Models Implementation

Dataset: Demographics.csv (Adult Income Dataset)

Objective: Implement and compare 6 classification models to predict income levels (<=50K or >50K)

Models Implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Naive Bayes Classifier (Gaussian)
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost
"""

# ==============================================================================
# 1. Import Required Libraries
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("All libraries imported successfully!")

# ==============================================================================
# 2. Load the Dataset
# ==============================================================================

print("Loading dataset...")
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df = pd.read_csv('demographics.csv', names=column_names, skipinitialspace=True)

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

# Replace '?' with NaN to identify missing values
print("\nReplacing '?' with NaN to identify missing values...")
df = df.replace('?', np.nan)

print("\nMissing values by column:")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0])
print(f"\nTotal rows with missing values: {df.isnull().any(axis=1).sum()}")

print("\nTarget variable distribution:")
print(df['income'].value_counts())

# ==============================================================================
# 3. Data Preprocessing
#
# This section handles:
# - Missing value treatment (replacing '?' with NaN and dropping rows)
# - Target variable encoding
# - Categorical variable encoding
# - Feature identification
# ==============================================================================

print("="*80)
print("DATA PREPROCESSING")
print("="*80)

# Handle missing values - drop rows with missing values
print(f"\nRows before dropping missing values: {len(df)}")
df = df.dropna()
print(f"Rows after dropping missing values: {len(df)}")

# Separate features and target
X = df.drop('income', axis=1)
y = df['income']

# Encode target variable
print("\nEncoding target variable...")
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
print(f"Classes: {label_encoder_y.classes_}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print("Preprocessing complete!")

# ==============================================================================
# 4. Train-Test Split and Feature Scaling
#
# Split the data into training (80%) and testing (20%) sets, then standardize 
# numerical features.
# ==============================================================================

# Split the data
print("Splitting data into train and test sets (80-20 split)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Standardize numerical features
print("\nStandardizing numerical features...")
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("Data preparation complete!")

# ==============================================================================
# 5. Model Evaluation Function
#
# Define a reusable function to train and evaluate models with comprehensive 
# metrics.
# ==============================================================================

# Function to evaluate models
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a classification model
    """
    print("\n" + "="*80)
    print(f"{model_name}")
    print("="*80)
    
    # Train the model
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Get prediction probabilities for AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = y_pred_test
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred_test)
    
    # Display metrics in tabular format
    print("\nPerformance Metrics:")
    print("-" * 80)
    metrics_df = pd.DataFrame({
        'Metric': ['Training Accuracy', 'Test Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score', 'MCC Score'],
        'Value': [train_accuracy, test_accuracy, auc, precision, recall, f1, mcc]
    })
    print(metrics_df.to_string(index=False))
    print("-" * 80)
    
    # Alternative formatted table view
    print("\nFormatted Metrics Table:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Value':>20}")
    print("-" * 50)
    print(f"{'Training Accuracy':<25} {train_accuracy:>20.4f}")
    print(f"{'Test Accuracy':<25} {test_accuracy:>20.4f}")
    print(f"{'AUC':<25} {auc:>20.4f}")
    print(f"{'Precision':<25} {precision:>20.4f}")
    print(f"{'Recall':<25} {recall:>20.4f}")
    print(f"{'F1-Score':<25} {f1:>20.4f}")
    print(f"{'MCC Score':<25} {mcc:>20.4f}")
    print("-" * 50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder_y.classes_))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(cm, 
                         index=[f'Actual {label_encoder_y.classes_[0]}', f'Actual {label_encoder_y.classes_[1]}'],
                         columns=[f'Predicted {label_encoder_y.classes_[0]}', f'Predicted {label_encoder_y.classes_[1]}'])
    print(cm_df)
    
    return {
        'ML Model Name': model_name,
        'Accuracy': test_accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc
    }

# Store results
results = []

print("Evaluation function defined successfully!")

# ==============================================================================
# 6. Model 1: Logistic Regression
#
# Logistic Regression is a linear model for binary classification that uses 
# the logistic function to predict probabilities.
# ==============================================================================

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_results = evaluate_model(lr_model, "LOGISTIC REGRESSION", 
                            X_train_scaled, X_test_scaled, y_train, y_test)
results.append(lr_results)

# ==============================================================================
# 7. Model 2: Decision Tree Classifier
#
# Decision Tree is a tree-structured model that makes decisions by splitting 
# data based on feature values.
# ==============================================================================

dt_model = DecisionTreeClassifier(random_state=42)
dt_results = evaluate_model(dt_model, "DECISION TREE CLASSIFIER", 
                            X_train, X_test, y_train, y_test)
results.append(dt_results)

# ==============================================================================
# 8. Model 3: K-Nearest Neighbors Classifier
#
# KNN classifies samples based on the majority class of their k nearest 
# neighbors in the feature space.
# ==============================================================================

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_results = evaluate_model(knn_model, "K-NEAREST NEIGHBORS CLASSIFIER", 
                             X_train_scaled, X_test_scaled, y_train, y_test)
results.append(knn_results)

# ==============================================================================
# 9. Model 4: Naive Bayes Classifier (Gaussian)
#
# Gaussian Naive Bayes assumes features follow a normal distribution and 
# applies Bayes' theorem with strong independence assumptions.
# ==============================================================================

nb_model = GaussianNB()
nb_results = evaluate_model(nb_model, "NAIVE BAYES CLASSIFIER (GAUSSIAN)", 
                            X_train_scaled, X_test_scaled, y_train, y_test)
results.append(nb_results)

# ==============================================================================
# 10. Model 5: Random Forest Classifier (Ensemble)
#
# Random Forest is an ensemble method that builds multiple decision trees and 
# combines their predictions through voting.
# ==============================================================================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_results = evaluate_model(rf_model, "RANDOM FOREST CLASSIFIER", 
                            X_train, X_test, y_train, y_test)
results.append(rf_results)

# ==============================================================================
# 11. Model 6: XGBoost Classifier (Ensemble)
#
# XGBoost is a gradient boosting ensemble method that builds trees sequentially, 
# each correcting errors from previous trees.
# ==============================================================================

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_results = evaluate_model(xgb_model, "XGBOOST CLASSIFIER", 
                             X_train, X_test, y_train, y_test)
results.append(xgb_results)

# ==============================================================================
# 12. Model Comparison Summary
#
# Compare all models based on their performance metrics to identify the best 
# performer.
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY - MODEL COMPARISON")
print("="*80)

# Create results dataframe
results_df = pd.DataFrame(results)

print("-" * 120)
print(f"{'ML Model Name':<40} {'Accuracy':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}")
print("-" * 120)
for _, row in results_df.iterrows():
    print(f"{row['ML Model Name']:<40} {row['Accuracy']:>10.4f} {row['AUC']:>10.4f} {row['Precision']:>10.4f} {row['Recall']:>10.4f} {row['F1']:>10.4f} {row['MCC']:>10.4f}")
print("-" * 120)

# Find best model based on test accuracy
best_model_idx = results_df['Accuracy'].idxmax()
best_model = results_df.loc[best_model_idx, 'ML Model Name']
best_accuracy = results_df.loc[best_model_idx, 'Accuracy']
best_auc = results_df.loc[best_model_idx, 'AUC']

print(f"\n{'='*80}")
print(f"Best Model: {best_model}")
print(f"Test Accuracy: {best_accuracy:.4f}")
print(f"AUC Score: {best_auc:.4f}")
print(f"{'='*80}")

# ==============================================================================
# 13. Model Performance Observations
#
# Detailed observations on the performance of each model on the Demographics 
# dataset.
# ==============================================================================

print("="*80)
print("MODEL PERFORMANCE OBSERVATIONS")
print("="*80)

# Sort models by accuracy for better comparison
results_sorted = results_df.sort_values('Accuracy', ascending=False)

print("\n1. OVERALL PERFORMANCE RANKING (by Accuracy):")
print("-" * 80)
for idx, (i, row) in enumerate(results_sorted.iterrows(), 1):
    print(f"{idx}. {row['ML Model Name']}: {row['Accuracy']:.4f}")

print("\n2. DETAILED OBSERVATIONS:")
print("-" * 80)

for _, row in results_df.iterrows():
    model_name = row['ML Model Name']
    accuracy = row['Accuracy']
    auc = row['AUC']
    precision = row['Precision']
    recall = row['Recall']
    f1 = row['F1']
    mcc = row['MCC']
    
    print(f"\n{model_name}:")
    
    # Performance level
    if accuracy >= 0.85:
        print(f"  ✓ Excellent performance with {accuracy:.2%} accuracy")
    elif accuracy >= 0.80:
        print(f"  ✓ Good performance with {accuracy:.2%} accuracy")
    elif accuracy >= 0.75:
        print(f"  ○ Moderate performance with {accuracy:.2%} accuracy")
    else:
        print(f"  ✗ Lower performance with {accuracy:.2%} accuracy")
    
    # AUC interpretation
    if auc >= 0.90:
        print(f"  ✓ Excellent discrimination capability (AUC: {auc:.4f})")
    elif auc >= 0.80:
        print(f"  ✓ Good discrimination capability (AUC: {auc:.4f})")
    else:
        print(f"  ○ Moderate discrimination capability (AUC: {auc:.4f})")
    
    # Precision-Recall balance
    if abs(precision - recall) < 0.02:
        print(f"  ✓ Well-balanced precision ({precision:.4f}) and recall ({recall:.4f})")
    elif precision > recall:
        print(f"  ○ Higher precision ({precision:.4f}) than recall ({recall:.4f}) - fewer false positives")
    else:
        print(f"  ○ Higher recall ({recall:.4f}) than precision ({precision:.4f}) - fewer false negatives")
    
    # F1 Score
    print(f"  • F1-Score: {f1:.4f} - Overall balance between precision and recall")
    
    # MCC interpretation
    if mcc >= 0.50:
        print(f"  ✓ Strong correlation (MCC: {mcc:.4f})")
    elif mcc >= 0.30:
        print(f"  ○ Moderate correlation (MCC: {mcc:.4f})")
    else:
        print(f"  ✗ Weak correlation (MCC: {mcc:.4f})")

print("\n3. KEY INSIGHTS:")
print("-" * 80)

# Best overall model
best_model_name = results_sorted.iloc[0]['ML Model Name']
best_accuracy = results_sorted.iloc[0]['Accuracy']
best_f1 = results_sorted.iloc[0]['F1']

print(f"\n• Best Overall Model: {best_model_name}")
print(f"  - Achieved highest accuracy of {best_accuracy:.2%}")
print(f"  - F1-Score: {best_f1:.4f}")

# Best AUC
best_auc_idx = results_df['AUC'].idxmax()
best_auc_model = results_df.loc[best_auc_idx, 'ML Model Name']
best_auc_score = results_df.loc[best_auc_idx, 'AUC']
print(f"\n• Best AUC Score: {best_auc_model} ({best_auc_score:.4f})")

# Best MCC
best_mcc_idx = results_df['MCC'].idxmax()
best_mcc_model = results_df.loc[best_mcc_idx, 'ML Model Name']
best_mcc_score = results_df.loc[best_mcc_idx, 'MCC']
print(f"\n• Best MCC Score: {best_mcc_model} ({best_mcc_score:.4f})")

# Ensemble vs Traditional comparison
ensemble_models = results_df[results_df['ML Model Name'].str.contains('FOREST|XGBOOST')]
traditional_models = results_df[~results_df['ML Model Name'].str.contains('FOREST|XGBOOST')]

if len(ensemble_models) > 0 and len(traditional_models) > 0:
    avg_ensemble_acc = ensemble_models['Accuracy'].mean()
    avg_traditional_acc = traditional_models['Accuracy'].mean()
    print(f"\n• Ensemble Models Average Accuracy: {avg_ensemble_acc:.2%}")
    print(f"• Traditional Models Average Accuracy: {avg_traditional_acc:.2%}")
    
    if avg_ensemble_acc > avg_traditional_acc:
        improvement = ((avg_ensemble_acc - avg_traditional_acc) / avg_traditional_acc) * 100
        print(f"  → Ensemble models outperform traditional models by {improvement:.1f}%")

# Worst performing model
worst_model_name = results_sorted.iloc[-1]['ML Model Name']
worst_accuracy = results_sorted.iloc[-1]['Accuracy']
print(f"\n• Lowest Performing Model: {worst_model_name} ({worst_accuracy:.2%})")

print("\n4. RECOMMENDATIONS:")
print("-" * 80)
print(f"\n• For deployment: Use {best_model_name} for best overall performance")
print(f"• For interpretability: Consider DECISION TREE or LOGISTIC REGRESSION")
print(f"• For handling imbalanced data: Consider models with high MCC scores")
if best_auc_score >= 0.85:
    print(f"• The {best_auc_model} shows excellent ability to distinguish between classes")

print("\n" + "="*80)
