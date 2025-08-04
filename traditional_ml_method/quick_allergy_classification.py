import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import re

print("Libraries loaded!")

# Load the processed data
try:
    combined_data = pd.read_excel('processed_combined_genus_metabolite203.xlsx', index_col=0)
    print(f"Data loaded: {combined_data.shape}")
except:
    print("Error loading data. Please run the data processing notebook first!")
    exit()

# Separate features and labels
X = combined_data.drop(['label', 'group'], axis=1)
y = combined_data['label']

print(f"Features: {X.shape}")
print(f"Labels: {y.value_counts()}")

# FIX: Clean column names for XGBoost
def clean_feature_names(df):
    """Clean feature names to be compatible with XGBoost"""
    new_columns = []
    for col in df.columns:
        # Replace problematic characters
        clean_col = re.sub(r'[<>\[\],]', '_', str(col))
        clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', clean_col)
        clean_col = re.sub(r'_+', '_', clean_col)  # Replace multiple underscores
        clean_col = clean_col.strip('_')  # Remove leading/trailing underscores
        new_columns.append(clean_col)
    
    df_clean = df.copy()
    df_clean.columns = new_columns
    return df_clean

# Clean the feature names
X_clean = clean_feature_names(X)
print(f"Feature names cleaned. Sample names: {list(X_clean.columns[:5])}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Define models with simpler parameters to avoid issues
models = {
    'XGBoost': XGBClassifier(
        n_estimators=50, 
        max_depth=4, 
        learning_rate=0.1, 
        random_state=42,
        eval_metric='logloss'  # Suppress warnings
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=50, 
        max_depth=4, 
        learning_rate=0.1, 
        random_state=42, 
        verbose=-1,
        force_col_wise=True  # Avoid threading issues
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=50, 
        max_depth=8, 
        random_state=42,
        n_jobs=1  # Avoid multiprocessing issues
    )
}

# Train and evaluate models
results = {}
print("\n=== MODEL EVALUATION ===")

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    try:
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')  # Reduced CV folds
        
        # Train on full training set and test
        model.fit(X_train, y_train)
        test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)
        
        results[name] = {
            'CV_AUC': cv_scores.mean(),
            'CV_std': cv_scores.std(),
            'Test_AUC': test_auc,
            'model': model
        }
        
        print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Test AUC: {test_auc:.3f}")
        
    except Exception as e:
        print(f"  Error with {name}: {str(e)}")
        continue

# Multi-Modal Approach
print("\n=== MULTI-MODAL APPROACH ===")

# Separate genus and metabolite features
genus_features = [col for col in X_clean.columns if 'genus' in col.lower()]
metabolite_features = [col for col in X_clean.columns if 'metabolite' in col.lower()]

print(f"Genus features: {len(genus_features)}")
print(f"Metabolite features: {len(metabolite_features)}")

if len(genus_features) > 0 and len(metabolite_features) > 0:
    # Train separate models
    genus_model = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric='logloss')
    metabolite_model = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric='logloss')
    
    genus_model.fit(X_train[genus_features], y_train)
    metabolite_model.fit(X_train[metabolite_features], y_train)
    
    # Get predictions
    genus_pred = genus_model.predict_proba(X_test[genus_features])[:, 1]
    metabolite_pred = metabolite_model.predict_proba(X_test[metabolite_features])[:, 1]
    
    # Simple ensemble - average predictions
    ensemble_pred = (genus_pred + metabolite_pred) / 2
    
    # Evaluate
    genus_auc = roc_auc_score(y_test, genus_pred)
    metabolite_auc = roc_auc_score(y_test, metabolite_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    
    print(f"Genus only AUC: {genus_auc:.3f}")
    print(f"Metabolite only AUC: {metabolite_auc:.3f}")
    print(f"Ensemble AUC: {ensemble_auc:.3f}")
    
    # Add to results
    results['Genus_Only'] = {'Test_AUC': genus_auc}
    results['Metabolite_Only'] = {'Test_AUC': metabolite_auc}
    results['Ensemble'] = {'Test_AUC': ensemble_auc}

# Feature Selection
print("\n=== FEATURE SELECTION ===")

n_features = min(30, X_clean.shape[1] // 3)  # Select fewer features to avoid overfitting
selector = SelectKBest(f_classif, k=n_features)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train model on selected features
model_selected = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric='logloss')
model_selected.fit(X_train_selected, y_train)

# Evaluate
pred_selected = model_selected.predict_proba(X_test_selected)[:, 1]
auc_selected = roc_auc_score(y_test, pred_selected)

print(f"Selected features AUC: {auc_selected:.3f}")
print(f"Number of features used: {n_features}")

results['Feature_Selected'] = {'Test_AUC': auc_selected}

# Show top features
feature_scores = selector.scores_
feature_names = X_clean.columns
top_features = pd.DataFrame({
    'feature': feature_names,
    'score': feature_scores
}).sort_values('score', ascending=False).head(10)

print("\nTop 10 features:")
print(top_features)

# Find best method
print("\n=== RESULTS SUMMARY ===")

test_aucs = {name: result['Test_AUC'] for name, result in results.items() if 'Test_AUC' in result}
best_method = max(test_aucs, key=test_aucs.get)
best_score = test_aucs[best_method]

print(f"Best method: {best_method}")
print(f"Best AUC: {best_score:.3f}")

print("\nAll results:")
for method, auc in sorted(test_aucs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {method}: {auc:.3f}")

# Plot results
plt.figure(figsize=(12, 6))
methods = list(test_aucs.keys())
scores = list(test_aucs.values())

bars = plt.bar(methods, scores, color='skyblue')
best_idx = methods.index(best_method)
bars[best_idx].set_color('orange')  # Highlight best

plt.title('Model Performance Comparison')
plt.ylabel('AUC Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Final evaluation with best method
print(f"\n=== FINAL MODEL EVALUATION ===")

if best_method == 'Ensemble' and 'Ensemble' in results:
    final_predictions = ensemble_pred
    print("Using ensemble of genus + metabolite models")
elif best_method == 'Feature_Selected':
    final_predictions = pred_selected
    print(f"Using feature selection with {n_features} features")
else:
    # Use the best single model
    if best_method in results and 'model' in results[best_method]:
        final_model = results[best_method]['model']
        final_predictions = final_model.predict_proba(X_test)[:, 1]
        print(f"Using {best_method}")
    else:
        # Fallback to XGBoost
        final_model = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric='logloss')
        final_model.fit(X_train, y_train)
        final_predictions = final_model.predict_proba(X_test)[:, 1]
        print("Using XGBoost as fallback")

# Convert probabilities to binary predictions
binary_predictions = (final_predictions > 0.5).astype(int)

# Detailed evaluation
print(f"\nFinal Model Performance:")
print(f"AUC: {roc_auc_score(y_test, final_predictions):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, binary_predictions, target_names=['N', 'IgE']))

# Confusion Matrix
cm = confusion_matrix(y_test, binary_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['N', 'IgE'], yticklabels=['N', 'IgE'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nModel successfully trained and evaluated!")
print(f"Best approach: {best_method} with AUC = {best_score:.3f}")
print("Results saved as 'model_comparison.png' and 'confusion_matrix.png'")