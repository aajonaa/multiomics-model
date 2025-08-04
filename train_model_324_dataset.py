#!/usr/bin/env python3
"""
训练324数据集的多模态分类器
使用预分割的train/test数据: 
- 324_X_train_source115_genus_metabolite.xlsx
- 324_X_test_source203_genus_metabolite.xlsx
- 324_y_train_source115_genus_metabolite.xlsx  
- 324_y_test_source203_genus_metabolite.xlsx
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

class EnhancedMultimodalClassifier(nn.Module):
    """增强的多模态分类器 - 注意力引导的特征融合"""
    def __init__(self, micro_input_dim, metab_input_dim, hidden_dim=128, dropout_rate=0.4):
        super(EnhancedMultimodalClassifier, self).__init__()
        
        self.micro_input_dim = micro_input_dim
        self.metab_input_dim = metab_input_dim
        
        # 微生物组路径
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
        
        # 代谢物路径
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
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
        # 最终分类器
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
        
    def forward(self, x_micro, x_metab):
        micro_features = self.micro_net(x_micro)
        metab_features = self.metab_net(x_metab)
        
        # 特征串联
        combined = torch.cat([micro_features, metab_features], dim=1)
        
        # 注意力权重
        attention_weights = self.attention(combined)
        micro_weight = attention_weights[:, 0:1]
        metab_weight = attention_weights[:, 1:2]
        
        # 加权组合
        weighted_features = torch.cat([
            micro_features * micro_weight,
            metab_features * metab_weight
        ], dim=1)
        
        # 分类
        output = self.classifier(weighted_features)
        
        return {
            'classification': output,
            'micro_features': micro_features,
            'metab_features': metab_features,
            'attention_weights': attention_weights,
            'combined_features': weighted_features
        }

def clean_column_names(df):
    """清理列名"""
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

def create_data_loader(X_micro, X_metab, y, batch_size=32, shuffle=True, sampler=None):
    """创建数据加载器"""
    X_micro_tensor = torch.FloatTensor(X_micro)
    X_metab_tensor = torch.FloatTensor(X_metab)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    dataset = TensorDataset(X_micro_tensor, X_metab_tensor, y_tensor)
    
    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def feature_selection(X_micro, X_metab, y, micro_feature_names, metab_feature_names, 
                     n_features_micro=50, n_features_metab=50):
    """特征选择"""
    print(f"从{len(micro_feature_names)}个genus特征中选择前{n_features_micro}个...")
    print(f"从{len(metab_feature_names)}个metabolite特征中选择前{n_features_metab}个...")
    
    # 微生物特征选择
    selector_micro = SelectKBest(mutual_info_classif, k=min(n_features_micro, X_micro.shape[1]))
    X_micro_selected = selector_micro.fit_transform(X_micro, y)
    selected_micro_names = [micro_feature_names[i] for i in selector_micro.get_support(indices=True)]
    
    # 代谢物特征选择
    selector_metab = SelectKBest(mutual_info_classif, k=min(n_features_metab, X_metab.shape[1]))
    X_metab_selected = selector_metab.fit_transform(X_metab, y)
    selected_metab_names = [metab_feature_names[i] for i in selector_metab.get_support(indices=True)]
    
    print(f"实际选择了{X_micro_selected.shape[1]}个genus特征和{X_metab_selected.shape[1]}个metabolite特征")
    
    return X_micro_selected, X_metab_selected, selected_micro_names, selected_metab_names, selector_micro, selector_metab

def plot_training_curves(train_losses, val_aucs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 验证AUC曲线
    ax2.plot(val_aucs, 'r-', label='Validation AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_324.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_results(y_true, y_pred, y_prob):
    """绘制结果分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    
    # 2. ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0,1].plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.3f})')
    axes[0,1].plot([0, 1], [0, 1], 'r--')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 预测概率分布
    axes[1,0].hist(y_prob[y_true == 0], bins=20, alpha=0.7, label='Class 0', color='blue')
    axes[1,0].hist(y_prob[y_true == 1], bins=20, alpha=0.7, label='Class 1', color='red')
    axes[1,0].set_xlabel('Predicted Probability')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Prediction Probability Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 预测vs真实标签散点图
    jitter = np.random.normal(0, 0.05, len(y_true))
    axes[1,1].scatter(y_true + jitter, y_prob, alpha=0.6, c=y_true, cmap='coolwarm')
    axes[1,1].axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
    axes[1,1].set_xlabel('True Label')
    axes[1,1].set_ylabel('Predicted Probability')
    axes[1,1].set_title('Predicted Probability vs True Label')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_results_324.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🧬 多模态分类器训练 - 324数据集")
    print("=" * 70)
    
    # 加载数据
    print("📊 加载数据...")
    try:
        X_train = pd.read_excel('324_X_train_source115_genus_metabolite.xlsx', index_col=0)
        y_train = pd.read_excel('324_y_train_source115_genus_metabolite.xlsx', index_col=0)
        X_test = pd.read_excel('324_X_test_source203_genus_metabolite.xlsx', index_col=0)
        y_test = pd.read_excel('324_y_test_source203_genus_metabolite.xlsx', index_col=0)
        
        print(f"✅ 数据加载成功")
        print(f"   训练集: {X_train.shape[0]}个样本, {X_train.shape[1]}个特征")
        print(f"   测试集: {X_test.shape[0]}个样本, {X_test.shape[1]}个特征")
        
    except FileNotFoundError as e:
        print(f"❌ 数据文件未找到: {e}")
        return
    
    # 清理列名
    X_train = clean_column_names(X_train)
    X_test = clean_column_names(X_test)
    
    # 提取标签
    y_train_values = y_train.iloc[:, 0].values
    y_test_values = y_test.iloc[:, 0].values
    
    print(f"📈 类别分布:")
    print(f"   训练集: {dict(zip(['Class 0', 'Class 1'], np.bincount(y_train_values)))}")
    print(f"   测试集: {dict(zip(['Class 0', 'Class 1'], np.bincount(y_test_values)))}")
    
    # 分离多模态特征
    genus_features = [col for col in X_train.columns if 'genus' in col.lower()]
    metabolite_features = [col for col in X_train.columns if 'metabolite' in col.lower()]
    
    X_train_genus = X_train[genus_features].values
    X_train_metab = X_train[metabolite_features].values
    X_test_genus = X_test[genus_features].values
    X_test_metab = X_test[metabolite_features].values
    
    print(f"🔬 特征分析:")
    print(f"   Genus特征: {len(genus_features)}")
    print(f"   Metabolite特征: {len(metabolite_features)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用设备: {device}")
    
    # 使用所有特征，不进行特征选择
    print("\n🎯 使用所有特征（不进行特征选择）...")
    X_train_genus_selected = X_train_genus
    X_train_metab_selected = X_train_metab
    X_test_genus_selected = X_test_genus
    X_test_metab_selected = X_test_metab
    selected_genus_names = genus_features
    selected_metab_names = metabolite_features
    
    print(f"✅ 使用完整特征集: {len(genus_features)}个genus + {len(metabolite_features)}个metabolite特征")
    
    # 数据标准化
    print("📊 数据标准化...")
    scaler_genus = StandardScaler()
    scaler_metab = StandardScaler()
    
    X_train_genus_scaled = scaler_genus.fit_transform(X_train_genus_selected)
    X_train_metab_scaled = scaler_metab.fit_transform(X_train_metab_selected)
    
    X_test_genus_scaled = scaler_genus.transform(X_test_genus_selected)
    X_test_metab_scaled = scaler_metab.transform(X_test_metab_selected)
    
    print(f"✅ 最终特征维度:")
    print(f"   Genus: {X_train_genus_scaled.shape[1]}")
    print(f"   Metabolite: {X_train_metab_scaled.shape[1]}")
    
    # 创建模型
    print("\n🤖 创建模型...")
    model = EnhancedMultimodalClassifier(
        micro_input_dim=X_train_genus_scaled.shape[1],
        metab_input_dim=X_train_metab_scaled.shape[1],
        hidden_dim=256,  # 增大隐藏层以处理更多特征
        dropout_rate=0.5  # 增加dropout防止过拟合
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 处理类别不平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_values), y=y_train_values)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"类别权重: {class_weight_dict}")
    
    # 加权采样器
    sample_weights = [class_weight_dict[label] for label in y_train_values]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # 数据加载器
    train_loader = create_data_loader(
        X_train_genus_scaled, X_train_metab_scaled, y_train_values,
        batch_size=16, sampler=sampler  # 减小batch size适应高维特征
    )
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.BCELoss(weight=torch.FloatTensor([class_weights[1]]).to(device))
    
    # 训练循环
    print("\\n🚀 开始训练...")
    train_losses = []
    val_aucs = []
    best_auc = 0
    
    model.train()
    for epoch in range(100):
        epoch_loss = 0
        num_batches = 0
        
        for x_genus, x_metab, labels in train_loader:
            x_genus, x_metab, labels = x_genus.to(device), x_metab.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_genus, x_metab)
            
            loss = criterion(outputs['classification'], labels)
            
            # L2正则化
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            total_loss = loss + 0.001 * l2_reg
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            X_test_genus_tensor = torch.FloatTensor(X_test_genus_scaled).to(device)
            X_test_metab_tensor = torch.FloatTensor(X_test_metab_scaled).to(device)
            
            outputs = model(X_test_genus_tensor, X_test_metab_tensor)
            test_pred = outputs['classification'].cpu().numpy().flatten()
            
            val_auc = roc_auc_score(y_test_values, test_pred)
            val_aucs.append(val_auc)
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), 'best_model_324.pth')
        
        model.train()
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Val_AUC={val_auc:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model_324.pth'))
    
    # 最终评估
    print("\\n" + "="*70)
    print("🎯 最终评估结果")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        X_test_genus_tensor = torch.FloatTensor(X_test_genus_scaled).to(device)
        X_test_metab_tensor = torch.FloatTensor(X_test_metab_scaled).to(device)
        
        outputs = model(X_test_genus_tensor, X_test_metab_tensor)
        test_pred_prob = outputs['classification'].cpu().numpy().flatten()
        test_pred_binary = (test_pred_prob > 0.5).astype(int)
        attention_weights = outputs['attention_weights'].cpu().numpy()
    
    # 性能指标
    test_auc = roc_auc_score(y_test_values, test_pred_prob)
    accuracy = (test_pred_binary == y_test_values).mean()
    
    print(f"🔥 测试集 AUC: {test_auc:.4f}")
    print(f"🔥 测试集 准确率: {accuracy:.4f}")
    print(f"🔥 最佳验证 AUC: {best_auc:.4f}")
    
    print("\\n📊 分类报告:")
    print(classification_report(y_test_values, test_pred_binary, target_names=['Class 0', 'Class 1']))
    
    # 注意力分析
    avg_genus_attention = attention_weights[:, 0].mean()
    avg_metab_attention = attention_weights[:, 1].mean()
    print(f"\\n🎯 注意力权重分析:")
    print(f"   平均Genus注意力权重: {avg_genus_attention:.4f}")
    print(f"   平均Metabolite注意力权重: {avg_metab_attention:.4f}")
    
    # 可视化
    print("\\n📈 生成可视化图表...")
    plot_training_curves(train_losses, val_aucs)
    plot_results(y_test_values, test_pred_binary, test_pred_prob)
    
    # 保存结果
    results_df = pd.DataFrame({
        'sample_id': X_test.index,
        'true_label': y_test_values,
        'predicted_prob': test_pred_prob,
        'predicted_label': test_pred_binary,
        'genus_attention': attention_weights[:, 0],
        'metabolite_attention': attention_weights[:, 1]
    })
    results_df.to_csv('prediction_results_324.csv', index=False)
    
    print("\\n✅ 训练完成!")
    print("📁 输出文件:")
    print("   • best_model_324.pth - 最佳模型")
    print("   • training_curves_324.png - 训练曲线")
    print("   • model_results_324.png - 结果分析")
    print("   • prediction_results_324.csv - 预测结果")

if __name__ == "__main__":
    main()
