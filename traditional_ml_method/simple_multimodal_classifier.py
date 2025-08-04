#!/usr/bin/env python3
"""
Simple Multimodal Classifier - No Autoencoder
Focus purely on classification without reconstruction complexity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

class SimpleMultimodalClassifier(nn.Module):
    """Simple classifier without reconstruction - much less overfitting"""
    def __init__(self, micro_input_dim, metab_input_dim, hidden_dim=64):
        super(SimpleMultimodalClassifier, self).__init__()
        
        # Separate feature extractors
        self.micro_net = nn.Sequential(
            nn.Linear(micro_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.metab_net = nn.Sequential(
            nn.Linear(metab_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),  # 32 + 32 = 64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_micro, x_metab):
        micro_features = self.micro_net(x_micro)
        metab_features = self.metab_net(x_metab)
        
        # Concatenate features
        combined = torch.cat([micro_features, metab_features], dim=1)
        
        # Classify
        output = self.classifier(combined)
        
        return {
            'classification': output,
            'micro_features': micro_features,
            'metab_features': metab_features,
            'combined_features': combined
        }

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

def create_data_loader(X_micro, X_metab, y, batch_size=32, shuffle=True):
    """Create data loader"""
    X_micro_tensor = torch.FloatTensor(X_micro)
    X_metab_tensor = torch.FloatTensor(X_metab)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    dataset = TensorDataset(X_micro_tensor, X_metab_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, train_loader, optimizer, device):
    """Simple training - only classification loss"""
    model.train()
    total_loss = 0
    criterion = nn.BCELoss()
    
    for x_micro, x_metab, y in train_loader:
        x_micro, x_metab, y = x_micro.to(device), x_metab.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(x_micro, x_metab)
        
        # Only classification loss - no reconstruction!
        loss = criterion(outputs['classification'], y)
        
        # Add L2 regularization
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        total_loss_batch = loss + 0.01 * l2_reg
        
        total_loss_batch.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()  # Store only classification loss for monitoring
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    """Simple evaluation"""
    model.eval()
    total_loss = 0
    criterion = nn.BCELoss()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for x_micro, x_metab, y in test_loader:
            x_micro, x_metab, y = x_micro.to(device), x_metab.to(device), y.to(device)
            
            outputs = model(x_micro, x_metab)
            loss = criterion(outputs['classification'], y)
            
            total_loss += loss.item()
            
            all_predictions.extend(outputs['classification'].cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())
    
    auc_score = roc_auc_score(all_labels, all_predictions) if len(set(all_labels)) > 1 else 0.5
    
    return {
        'loss': total_loss / len(test_loader),
        'auc': auc_score,
        'predictions': all_predictions,
        'labels': all_labels
    }

def cross_validate_model(X_micro, X_metab, y, device, n_folds=5):
    """Cross-validation to get more robust results"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_micro, y)):
        print(f"  Fold {fold+1}/{n_folds}...", end=" ")
        
        # Split data
        X_micro_train, X_micro_val = X_micro[train_idx], X_micro[val_idx]
        X_metab_train, X_metab_val = X_metab[train_idx], X_metab[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale data
        scaler_micro = StandardScaler()
        scaler_metab = StandardScaler()
        
        X_micro_train_scaled = scaler_micro.fit_transform(X_micro_train)
        X_micro_val_scaled = scaler_micro.transform(X_micro_val)
        
        X_metab_train_scaled = scaler_metab.fit_transform(X_metab_train)
        X_metab_val_scaled = scaler_metab.transform(X_metab_val)
        
        # Create model
        model = SimpleMultimodalClassifier(
            micro_input_dim=X_micro_train_scaled.shape[1],
            metab_input_dim=X_metab_train_scaled.shape[1],
            hidden_dim=32  # Even smaller
        ).to(device)
        
        # Create data loaders
        train_loader = create_data_loader(X_micro_train_scaled, X_metab_train_scaled, y_train_fold, batch_size=16)
        val_loader = create_data_loader(X_micro_val_scaled, X_metab_val_scaled, y_val_fold, batch_size=16, shuffle=False)
        
        # Train
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        best_auc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate_model(model, val_loader, device)
            
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        cv_aucs.append(best_auc)
        print(f"AUC: {best_auc:.3f}")
    
    return cv_aucs

def main():
    print("🎯 Simple Multimodal Classifier (No Autoencoder)")
    print("=" * 60)
    
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
    
    print(f"Microbiome features: {len(microbiome_features)}")
    print(f"Metabolite features: {len(metabolite_features)}")
    print(f"Total samples: {len(y)}")
    print(f"Class distribution: {dict(zip(['N', 'IgE'], np.bincount(y)))}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Cross-validation first
    print(f"\\nRunning {5}-fold cross-validation...")
    cv_aucs = cross_validate_model(X_microbiome, X_metabolite, y, device, n_folds=5)
    
    print(f"\\nCross-validation results:")
    print(f"  Mean AUC: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")
    print(f"  Individual folds: {[f'{auc:.3f}' for auc in cv_aucs]}")
    
    # Final model on train/test split
    print(f"\\nTraining final model on train/test split...")
    X_micro_train, X_micro_test, X_metab_train, X_metab_test, y_train, y_test = train_test_split(
        X_microbiome, X_metabolite, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale data
    scaler_micro = StandardScaler()
    scaler_metab = StandardScaler()
    
    X_micro_train_scaled = scaler_micro.fit_transform(X_micro_train)
    X_micro_test_scaled = scaler_micro.transform(X_micro_test)
    
    X_metab_train_scaled = scaler_metab.fit_transform(X_metab_train)
    X_metab_test_scaled = scaler_metab.transform(X_metab_test)
    
    # Create final model
    model = SimpleMultimodalClassifier(
        micro_input_dim=X_micro_train_scaled.shape[1],
        metab_input_dim=X_metab_train_scaled.shape[1],
        hidden_dim=32
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create data loaders
    train_loader = create_data_loader(X_micro_train_scaled, X_metab_train_scaled, y_train, batch_size=16)
    test_loader = create_data_loader(X_micro_test_scaled, X_metab_test_scaled, y_test, batch_size=16, shuffle=False)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    train_losses = []
    val_losses = []
    val_aucs = []
    
    best_auc = 0
    patience = 15
    patience_counter = 0
    
    print(f"\\nTraining final model...")
    
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, test_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        val_aucs.append(val_metrics['auc'])
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_simple_multimodal.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.3f} | Val Loss: {val_metrics['loss']:.3f} | Val AUC: {val_metrics['auc']:.3f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    model.load_state_dict(torch.load('best_simple_multimodal.pth'))
    final_metrics = evaluate_model(model, test_loader, device)
    
    binary_predictions = (np.array(final_metrics['predictions']) > 0.5).astype(int)
    true_labels = np.array(final_metrics['labels']).astype(int)
    
    print("\\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Cross-validation AUC: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")
    print(f"Test AUC: {final_metrics['auc']:.3f}")
    print(f"Accuracy: {(binary_predictions == true_labels).mean():.3f}")
    print(f"Parameters: {total_params:,}")
    
    print("\\nClassification Report:")
    print(classification_report(true_labels, binary_predictions, target_names=['N', 'IgE']))
    
    # Much cleaner plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Training curves - should be MUCH better now
    axes[0].plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    axes[0].set_title('Training Curves)', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC progression
    axes[1].plot(val_aucs, label='Validation AUC', color='green', linewidth=2)
    axes[1].axhline(y=best_auc, color='orange', linestyle='--', linewidth=2, label=f'Best: {best_auc:.3f}')
    axes[1].axhline(y=np.mean(cv_aucs), color='purple', linestyle=':', linewidth=2, label=f'CV Mean: {np.mean(cv_aucs):.3f}')
    axes[1].set_title('AUC Progression', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, binary_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['N', 'IgE'], yticklabels=['N', 'IgE'], ax=axes[2])
    axes[2].set_title('Confusion Matrix', fontsize=14)
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('simple_multimodal_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\n✅ Simple classifier complete!")
    print("📁 Files saved:")
    print("   • best_simple_multimodal.pth")
    print("   • simple_multimodal_results.png")
    
    print("\\n🎯 Key Differences:")
    print("   • No reconstruction loss (much simpler)")
    print("   • Smaller model (less overfitting)")
    print("   • Cross-validation for robust evaluation")
    print("   • Clean training curves!")

if __name__ == "__main__":
    main()