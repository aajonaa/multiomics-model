#!/usr/bin/env python3
"""
Enhanced Multimodal Classifier with Feature Importance Analysis
- Multiple feature importance methods
- Biological interpretation
- Visualization of important features
- SHAP-like analysis for neural networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

class EnhancedMultimodalClassifier(nn.Module):
    """Enhanced classifier with feature importance tracking"""
    def __init__(self, micro_input_dim, metab_input_dim, hidden_dim=128, dropout_rate=0.4):
        super(EnhancedMultimodalClassifier, self).__init__()
        
        self.micro_input_dim = micro_input_dim
        self.metab_input_dim = metab_input_dim
        
        # Microbiome pathway
        self.micro_net = nn.Sequential(
            nn.Linear(micro_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Metabolite pathway
        self.metab_net = nn.Sequential(
            nn.Linear(metab_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Attention mechanism for interpretability
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_micro, x_metab, return_attention=False):
        micro_features = self.micro_net(x_micro)
        metab_features = self.metab_net(x_metab)
        
        # Concatenate features
        combined = torch.cat([micro_features, metab_features], dim=1)
        
        # Apply attention
        attention_weights = self.attention(combined)
        micro_weight = attention_weights[:, 0:1]
        metab_weight = attention_weights[:, 1:2]
        
        # Weighted combination
        weighted_features = torch.cat([
            micro_features * micro_weight,
            metab_features * metab_weight
        ], dim=1)
        
        # Classify
        output = self.classifier(weighted_features)
        
        result = {
            'classification': output,
            'micro_features': micro_features,
            'metab_features': metab_features,
            'attention_weights': attention_weights,
            'combined_features': weighted_features
        }
        
        return result

class FeatureImportanceAnalyzer:
    """Comprehensive feature importance analysis for neural networks"""
    
    def __init__(self, model, X_micro, X_metab, y, micro_feature_names, metab_feature_names, device):
        self.model = model
        self.X_micro = X_micro
        self.X_metab = X_metab
        self.y = y
        self.micro_feature_names = micro_feature_names
        self.metab_feature_names = metab_feature_names
        self.device = device
        
    def gradient_based_importance(self):
        """Calculate feature importance using gradients"""
        print("Computing gradient-based feature importance...")
        
        self.model.eval()
        
        X_micro_tensor = torch.FloatTensor(self.X_micro).to(self.device)
        X_metab_tensor = torch.FloatTensor(self.X_metab).to(self.device)
        
        X_micro_tensor.requires_grad_(True)
        X_metab_tensor.requires_grad_(True)
        
        outputs = self.model(X_micro_tensor, X_metab_tensor)
        predictions = outputs['classification']
        
        # Calculate gradients
        gradients_micro = torch.autograd.grad(
            outputs=predictions.sum(),
            inputs=X_micro_tensor,
            create_graph=False,
            retain_graph=False
        )[0]
        
        gradients_metab = torch.autograd.grad(
            outputs=predictions.sum(),
            inputs=X_metab_tensor,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Feature importance = mean absolute gradient
        micro_importance = torch.abs(gradients_micro).mean(dim=0).cpu().numpy()
        metab_importance = torch.abs(gradients_metab).mean(dim=0).cpu().numpy()
        
        return micro_importance, metab_importance
    
    def permutation_importance(self, n_repeats=10):
        """Calculate permutation importance"""
        print("Computing permutation importance...")
        
        def predict_fn(X_micro, X_metab):
            self.model.eval()
            with torch.no_grad():
                X_micro_tensor = torch.FloatTensor(X_micro).to(self.device)
                X_metab_tensor = torch.FloatTensor(X_metab).to(self.device)
                outputs = self.model(X_micro_tensor, X_metab_tensor)
                return outputs['classification'].cpu().numpy().flatten()
        
        # Baseline performance
        baseline_pred = predict_fn(self.X_micro, self.X_metab)
        baseline_auc = roc_auc_score(self.y, baseline_pred)
        
        micro_importance = np.zeros(self.X_micro.shape[1])
        metab_importance = np.zeros(self.X_metab.shape[1])
        
        # Permute microbiome features
        for i in range(self.X_micro.shape[1]):
            importance_scores = []
            for _ in range(n_repeats):
                X_micro_perm = self.X_micro.copy()
                np.random.shuffle(X_micro_perm[:, i])  # Permute feature i
                
                perm_pred = predict_fn(X_micro_perm, self.X_metab)
                perm_auc = roc_auc_score(self.y, perm_pred)
                importance_scores.append(baseline_auc - perm_auc)
            
            micro_importance[i] = np.mean(importance_scores)
        
        # Permute metabolite features
        for i in range(self.X_metab.shape[1]):
            importance_scores = []
            for _ in range(n_repeats):
                X_metab_perm = self.X_metab.copy()
                np.random.shuffle(X_metab_perm[:, i])  # Permute feature i
                
                perm_pred = predict_fn(self.X_micro, X_metab_perm)
                perm_auc = roc_auc_score(self.y, perm_pred)
                importance_scores.append(baseline_auc - perm_auc)
            
            metab_importance[i] = np.mean(importance_scores)
        
        return micro_importance, metab_importance
    
    def attention_importance(self):
        """Extract attention weights as feature importance"""
        print("Computing attention-based importance...")
        
        self.model.eval()
        with torch.no_grad():
            X_micro_tensor = torch.FloatTensor(self.X_micro).to(self.device)
            X_metab_tensor = torch.FloatTensor(self.X_metab).to(self.device)
            
            outputs = self.model(X_micro_tensor, X_metab_tensor)
            attention_weights = outputs['attention_weights'].cpu().numpy()
            
            # Average attention weights across samples
            avg_micro_attention = attention_weights[:, 0].mean()
            avg_metab_attention = attention_weights[:, 1].mean()
            
            return avg_micro_attention, avg_metab_attention, attention_weights
    
    def integrated_gradients(self, n_steps=50):
        """Integrated gradients for feature attribution"""
        print("Computing integrated gradients...")
        
        self.model.eval()
        
        # Baseline (zeros)
        baseline_micro = torch.zeros_like(torch.FloatTensor(self.X_micro)).to(self.device)
        baseline_metab = torch.zeros_like(torch.FloatTensor(self.X_metab)).to(self.device)
        
        X_micro_tensor = torch.FloatTensor(self.X_micro).to(self.device)
        X_metab_tensor = torch.FloatTensor(self.X_metab).to(self.device)
        
        # Integrated gradients
        integrated_grads_micro = torch.zeros_like(X_micro_tensor)
        integrated_grads_metab = torch.zeros_like(X_metab_tensor)
        
        for step in range(n_steps):
            alpha = step / n_steps
            
            # Interpolated inputs
            interp_micro = baseline_micro + alpha * (X_micro_tensor - baseline_micro)
            interp_metab = baseline_metab + alpha * (X_metab_tensor - baseline_metab)
            
            interp_micro.requires_grad_(True)
            interp_metab.requires_grad_(True)
            
            outputs = self.model(interp_micro, interp_metab)
            predictions = outputs['classification']
            
            # Calculate gradients
            grads_micro = torch.autograd.grad(
                outputs=predictions.sum(),
                inputs=interp_micro,
                create_graph=False,
                retain_graph=False
            )[0]
            
            grads_metab = torch.autograd.grad(
                outputs=predictions.sum(),
                inputs=interp_metab,
                create_graph=False,
                retain_graph=False
            )[0]
            
            integrated_grads_micro += grads_micro / n_steps
            integrated_grads_metab += grads_metab / n_steps
        
        # Scale by input difference
        integrated_grads_micro *= (X_micro_tensor - baseline_micro)
        integrated_grads_metab *= (X_metab_tensor - baseline_metab)
        
        # Average across samples
        micro_importance = torch.abs(integrated_grads_micro).mean(dim=0).cpu().numpy()
        metab_importance = torch.abs(integrated_grads_metab).mean(dim=0).cpu().numpy()
        
        return micro_importance, metab_importance
    
    def analyze_all_methods(self):
        """Run all feature importance methods"""
        results = {}
        
        # 1. Gradient-based
        try:
            grad_micro, grad_metab = self.gradient_based_importance()
            results['gradient'] = {'micro': grad_micro, 'metab': grad_metab}
        except Exception as e:
            print(f"Gradient method failed: {e}")
        
        # 2. Permutation importance
        try:
            perm_micro, perm_metab = self.permutation_importance(n_repeats=5)
            results['permutation'] = {'micro': perm_micro, 'metab': perm_metab}
        except Exception as e:
            print(f"Permutation method failed: {e}")
        
        # 3. Attention weights
        try:
            avg_micro_att, avg_metab_att, attention_weights = self.attention_importance()
            results['attention'] = {
                'micro_avg': avg_micro_att, 
                'metab_avg': avg_metab_att,
                'weights': attention_weights
            }
        except Exception as e:
            print(f"Attention method failed: {e}")
        
        # 4. Integrated gradients
        try:
            ig_micro, ig_metab = self.integrated_gradients(n_steps=20)
            results['integrated_gradients'] = {'micro': ig_micro, 'metab': ig_metab}
        except Exception as e:
            print(f"Integrated gradients failed: {e}")
        
        return results

def clean_column_names(df):
    """Clean column names"""
    df = df.copy()
    df.columns = df.columns.str.replace('[', '', regex=False)
    df.columns = df.columns.str.replace(']', '', regex=False)
    df.columns = df.columns.str.replace('<', '_lt_', regex=False)
    df.columns = df.columns.str.replace('>', '_gt_', regex=False)
    df.columns = df.columns.str.replace('(', '_', regex=False)
    df.columns = df.columns.str.replace(')', '_', regex=False)
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    df.columns = df.columns.str.replace('-', '_', regex=False)
    df.columns = df.columns.str.replace('/', '_', regex=False)
    df.columns = df.columns.str.replace('\\', '_', regex=False)
    df.columns = df.columns.str.replace('_+', '_', regex=True)
    df.columns = df.columns.str.strip('_')
    return df

def create_weighted_data_loader(X_micro, X_metab, y, batch_size=32, sampler=None):
    """Create data loader with optional weighted sampling"""
    X_micro_tensor = torch.FloatTensor(X_micro)
    X_metab_tensor = torch.FloatTensor(X_metab)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    dataset = TensorDataset(X_micro_tensor, X_metab_tensor, y_tensor)
    
    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def feature_selection(X_micro, X_metab, y, micro_feature_names, metab_feature_names, n_features_micro=50, n_features_metab=50):
    """Select most informative features and return names"""
    print(f"Selecting top {n_features_micro} microbiome and {n_features_metab} metabolite features...")
    
    # Microbiome feature selection
    selector_micro = SelectKBest(mutual_info_classif, k=min(n_features_micro, X_micro.shape[1]))
    X_micro_selected = selector_micro.fit_transform(X_micro, y)
    selected_micro_names = [micro_feature_names[i] for i in selector_micro.get_support(indices=True)]
    
    # Metabolite feature selection
    selector_metab = SelectKBest(mutual_info_classif, k=min(n_features_metab, X_metab.shape[1]))
    X_metab_selected = selector_metab.fit_transform(X_metab, y)
    selected_metab_names = [metab_feature_names[i] for i in selector_metab.get_support(indices=True)]
    
    print(f"Selected {X_micro_selected.shape[1]} microbiome and {X_metab_selected.shape[1]} metabolite features")
    
    return X_micro_selected, X_metab_selected, selected_micro_names, selected_metab_names, selector_micro, selector_metab

def visualize_feature_importance(importance_results, micro_names, metab_names, top_n=15):
    """Create comprehensive feature importance visualizations"""
    
    n_methods = len([k for k in importance_results.keys() if k != 'attention'])
    fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 12))
    
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    method_idx = 0
    
    for method_name, results in importance_results.items():
        if method_name == 'attention':
            continue
            
        micro_imp = results['micro']
        metab_imp = results['metab']
        
        # Top microbiome features
        micro_df = pd.DataFrame({
            'feature': micro_names,
            'importance': micro_imp
        }).sort_values('importance', ascending=False).head(top_n)
        
        axes[0, method_idx].barh(range(len(micro_df)), micro_df['importance'], color='lightblue')
        axes[0, method_idx].set_yticks(range(len(micro_df)))
        axes[0, method_idx].set_yticklabels([name.replace('genus_', '') for name in micro_df['feature']], fontsize=10)
        axes[0, method_idx].set_xlabel('Importance Score')
        axes[0, method_idx].set_title(f'Top Microbiome Features ({method_name.title()})')
        axes[0, method_idx].grid(axis='x', alpha=0.3)
        
        # Top metabolite features
        metab_df = pd.DataFrame({
            'feature': metab_names,
            'importance': metab_imp
        }).sort_values('importance', ascending=False).head(top_n)
        
        axes[1, method_idx].barh(range(len(metab_df)), metab_df['importance'], color='lightcoral')
        axes[1, method_idx].set_yticks(range(len(metab_df)))
        axes[1, method_idx].set_yticklabels([name.replace('metabolite_', '') for name in metab_df['feature']], fontsize=10)
        axes[1, method_idx].set_xlabel('Importance Score')
        axes[1, method_idx].set_title(f'Top Metabolite Features ({method_name.title()})')
        axes[1, method_idx].grid(axis='x', alpha=0.3)
        
        method_idx += 1
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Attention analysis if available
    if 'attention' in importance_results:
        attention_data = importance_results['attention']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Attention weights distribution
        weights = attention_data['weights']
        axes[0].hist(weights[:, 0], bins=20, alpha=0.7, label='Microbiome Attention', color='blue')
        axes[0].hist(weights[:, 1], bins=20, alpha=0.7, label='Metabolite Attention', color='red')
        axes[0].axvline(x=attention_data['micro_avg'], color='blue', linestyle='--', 
                       label=f'Micro Avg: {attention_data["micro_avg"]:.3f}')
        axes[0].axvline(x=attention_data['metab_avg'], color='red', linestyle='--', 
                       label=f'Metab Avg: {attention_data["metab_avg"]:.3f}')
        axes[0].set_xlabel('Attention Weight')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Attention Weight Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Attention correlation with predictions
        # This would require predictions per sample - simplified version
        avg_weights = np.mean(weights, axis=0)
        modalities = ['Microbiome', 'Metabolite']
        colors = ['blue', 'red']
        
        bars = axes[1].bar(modalities, avg_weights, color=colors, alpha=0.7)
        axes[1].set_ylabel('Average Attention Weight')
        axes[1].set_title('Average Attention by Modality')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, weight in zip(bars, avg_weights):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{weight:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('attention_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🔍 Enhanced Multimodal Classifier with Feature Importance")
    print("=" * 70)
    
    # Load data
    try:
        combined_data = pd.read_excel('processed_combined_genus_metabolite203.xlsx', index_col=0)
        print(f"Data loaded: {combined_data.shape}")
    except FileNotFoundError:
        print("Error: Please run the data processing notebook first!")
        return
    
    combined_data = clean_column_names(combined_data)
    
    X = combined_data.drop(['label', 'group'], axis=1)
    y = combined_data['label'].values
    
    # Separate modalities
    microbiome_features = [col for col in X.columns if col.startswith('genus_')]
    metabolite_features = [col for col in X.columns if col.startswith('metabolite_')]
    
    X_microbiome = X[microbiome_features].values
    X_metabolite = X[metabolite_features].values
    
    print(f"Original features - Microbiome: {len(microbiome_features)}, Metabolite: {len(metabolite_features)}")
    print(f"Class distribution: {dict(zip(['N', 'IgE'], np.bincount(y)))}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Feature selection
    X_micro_selected, X_metab_selected, selected_micro_names, selected_metab_names, selector_micro, selector_metab = feature_selection(
        X_microbiome, X_metabolite, y, microbiome_features, metabolite_features, 
        n_features_micro=30, n_features_metab=30
    )
    
    # Train/test split
    X_micro_train, X_micro_test, X_metab_train, X_metab_test, y_train, y_test = train_test_split(
        X_micro_selected, X_metab_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale data
    scaler_micro = StandardScaler()
    scaler_metab = StandardScaler()
    
    X_micro_train_scaled = scaler_micro.fit_transform(X_micro_train)
    X_micro_test_scaled = scaler_micro.transform(X_micro_test)
    
    X_metab_train_scaled = scaler_metab.fit_transform(X_metab_train)
    X_metab_test_scaled = scaler_metab.transform(X_metab_test)
    
    print(f"\\nTraining samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    
    # Train model
    print(f"\\nTraining model...")
    model = EnhancedMultimodalClassifier(
        micro_input_dim=X_micro_train_scaled.shape[1],
        metab_input_dim=X_metab_train_scaled.shape[1],
        hidden_dim=96,
        dropout_rate=0.4
    ).to(device)
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # Weighted sampler
    sample_weights = [class_weight_dict[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Data loader
    train_loader = create_weighted_data_loader(
        X_micro_train_scaled, X_metab_train_scaled, y_train, 
        batch_size=24, sampler=sampler
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    # Training loop
    criterion = nn.BCELoss(weight=torch.FloatTensor([class_weights[1]]).to(device))
    
    model.train()
    for epoch in range(80):
        total_loss = 0
        for x_micro, x_metab, labels in train_loader:
            x_micro, x_metab, labels = x_micro.to(device), x_metab.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_micro, x_metab)
            
            loss = criterion(outputs['classification'], labels)
            
            # L2 regularization
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            total_loss_batch = loss + 0.001 * l2_reg
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.3f}")
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        X_test_micro_tensor = torch.FloatTensor(X_micro_test_scaled).to(device)
        X_test_metab_tensor = torch.FloatTensor(X_metab_test_scaled).to(device)
        
        outputs = model(X_test_micro_tensor, X_test_metab_tensor)
        test_pred = outputs['classification'].cpu().numpy().flatten()
    
    test_auc = roc_auc_score(y_test, test_pred)
    binary_predictions = (test_pred > 0.5).astype(int)
    accuracy = (binary_predictions == y_test).mean()
    
    print("\\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"Test AUC: {test_auc:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    print("\\nClassification Report:")
    print(classification_report(y_test, binary_predictions, target_names=['N', 'IgE']))
    
    # Feature Importance Analysis
    print("\\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    analyzer = FeatureImportanceAnalyzer(
        model, X_micro_test_scaled, X_metab_test_scaled, y_test,
        selected_micro_names, selected_metab_names, device
    )
    
    importance_results = analyzer.analyze_all_methods()
    
    # Print top features for each method
    for method_name, results in importance_results.items():
        if method_name == 'attention':
            print(f"\\n{method_name.upper()} ANALYSIS:")
            print(f"  Average Microbiome Attention: {results['micro_avg']:.3f}")
            print(f"  Average Metabolite Attention: {results['metab_avg']:.3f}")
            continue
            
        print(f"\\n{method_name.upper()} IMPORTANCE:")
        
        # Top microbiome features
        micro_imp = results['micro']
        top_micro_idx = np.argsort(micro_imp)[-5:][::-1]
        print("  Top 5 Microbiome Features:")
        for i, idx in enumerate(top_micro_idx):
            feature_name = selected_micro_names[idx].replace('genus_', '')
            print(f"    {i+1}. {feature_name}: {micro_imp[idx]:.4f}")
        
        # Top metabolite features
        metab_imp = results['metab']
        top_metab_idx = np.argsort(metab_imp)[-5:][::-1]
        print("  Top 5 Metabolite Features:")
        for i, idx in enumerate(top_metab_idx):
            feature_name = selected_metab_names[idx].replace('metabolite_', '')
            print(f"    {i+1}. {feature_name}: {metab_imp[idx]:.4f}")
    
    # Create visualizations
    print(f"\\nCreating feature importance visualizations...")
    visualize_feature_importance(importance_results, selected_micro_names, selected_metab_names, top_n=15)
    
    # Save feature importance results
    importance_df_list = []
    for method_name, results in importance_results.items():
        if method_name == 'attention':
            continue
            
        # Microbiome features
        for i, (name, imp) in enumerate(zip(selected_micro_names, results['micro'])):
            importance_df_list.append({
                'method': method_name,
                'modality': 'microbiome',
                'feature': name,
                'importance': imp,
                'rank': np.argsort(results['micro'])[::-1].tolist().index(i) + 1
            })
        
        # Metabolite features
        for i, (name, imp) in enumerate(zip(selected_metab_names, results['metab'])):
            importance_df_list.append({
                'method': method_name,
                'modality': 'metabolite',
                'feature': name,
                'importance': imp,
                'rank': np.argsort(results['metab'])[::-1].tolist().index(i) + 1
            })
    
    importance_df = pd.DataFrame(importance_df_list)
    importance_df.to_csv('feature_importance_results.csv', index=False)
    
    print("\\n✅ Analysis complete!")
    print("📁 Files saved:")
    print("   • feature_importance_analysis.png")
    print("   • attention_analysis.png")
    print("   • feature_importance_results.csv")
    
    print("\\n🔍 Key Insights:")
    print("   • Multiple importance methods provide robust feature ranking")
    print("   • Attention weights show modality importance")
    print("   • Permutation importance shows predictive value")
    print("   • Gradient methods show model sensitivity")
    print("   • Results saved for biological interpretation")

if __name__ == "__main__":
    main()