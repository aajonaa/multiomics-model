#!/usr/bin/env python3
"""
训练324数据集的多模态分类器 - 过敏性疾病分类
使用预分割的train/test数据: 
- 324_X_train_source115_genus_metabolite.xlsx (训练集)
- 324_X_test_source203_genus_metabolite.xlsx (测试集)
- 324_y_train_source115_genus_metabolite.xlsx (训练标签)
- 324_y_test_source203_genus_metabolite.xlsx (测试标签        # X_train = pd.read_excel('324_X_train_source115_genus_metabolite.xlsx', index_col=0)
        # y_train = pd.read_excel('324_y_train_source115_genus_metabolite.xlsx', index_col=0)
        # X_test = pd.read_excel('324_X_test_source203_genus_metabolite.xlsx', index_col=0)
        # y_test = pd.read_excel('324_y_test_source203_genus_metabolite.xlsx', index_col=0)

        X_train = pd.read_excel('324_X_test_source203_genus_metabolite.xlsx', index_col=0)
        y_train = pd.read_excel('324_y_test_source203_genus_metabolite.xlsx', index_col=0)
        X_test = pd.read_excel('324_X_train_source115_genus_metabolite.xlsx', index_col=0)
        y_test = pd.read_excel('324_y_train_source115_genus_metabolite.xlsx', index_col=0)
标签含义:
- 1 (IgE): 引起过敏反应的样本
- 0 (N group): 非过敏性样本（正常对照组）

特征: 使用所有891个特征 (567个genus + 324个metabolite)
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
        
        # 微生物组路径 (Genus features pathway)
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
        
        # 代谢物路径 (Metabolite features pathway)
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
        
        # 注意力机制 (Attention mechanism for interpretability)
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
        # 最终分类器 (Final classifier)
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
        # 处理微生物组特征
        micro_features = self.micro_net(x_micro)
        # 处理代谢物特征
        metab_features = self.metab_net(x_metab)
        
        # 特征串联
        combined = torch.cat([micro_features, metab_features], dim=1)
        
        # 计算注意力权重
        attention_weights = self.attention(combined)
        micro_weight = attention_weights[:, 0:1]
        metab_weight = attention_weights[:, 1:2]
        
        # 加权特征组合
        weighted_features = torch.cat([
            micro_features * micro_weight,
            metab_features * metab_weight
        ], dim=1)
        
        # 最终分类
        output = self.classifier(weighted_features)
        
        return {
            'classification': output,
            'micro_features': micro_features,
            'metab_features': metab_features,
            'attention_weights': attention_weights,
            'combined_features': weighted_features
        }

def clean_column_names(df):
    """清理列名，移除特殊字符"""
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
    """创建PyTorch数据加载器"""
    X_micro_tensor = torch.FloatTensor(X_micro)
    X_metab_tensor = torch.FloatTensor(X_metab)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    dataset = TensorDataset(X_micro_tensor, X_metab_tensor, y_tensor)
    
    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def plot_training_curves(train_losses, val_aucs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 验证AUC曲线
    ax2.plot(val_aucs, 'r-', label='Validation AUC', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_324_allergy.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_results(y_true, y_pred, y_prob, attention_weights):
    """绘制详细的结果分析图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=['Non-allergenic (N)', 'Allergenic (IgE)'],
                yticklabels=['Non-allergenic (N)', 'Allergenic (IgE)'])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    
    # 2. ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0,1].plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.3f})', linewidth=2)
    axes[0,1].plot([0, 1], [0, 1], 'r--', alpha=0.7)
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve - Allergy Classification')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 预测概率分布
    axes[0,2].hist(y_prob[y_true == 0], bins=20, alpha=0.7, label='Non-allergenic (N)', color='blue')
    axes[0,2].hist(y_prob[y_true == 1], bins=20, alpha=0.7, label='Allergenic (IgE)', color='red')
    axes[0,2].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
    axes[0,2].set_xlabel('Predicted Probability')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].set_title('Prediction Probability Distribution')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. 注意力权重分析
    avg_genus_attention = attention_weights[:, 0].mean()
    avg_metab_attention = attention_weights[:, 1].mean()
    modalities = ['Genus\\n(Microbiome)', 'Metabolite']
    avg_weights = [avg_genus_attention, avg_metab_attention]
    colors = ['lightblue', 'lightcoral']
    
    bars = axes[1,0].bar(modalities, avg_weights, color=colors, alpha=0.8)
    axes[1,0].set_ylabel('Average Attention Weight')
    axes[1,0].set_title('Average Attention by Modality')
    axes[1,0].set_ylim(0, 1)
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, weight in zip(bars, avg_weights):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                      f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. 注意力权重分布
    axes[1,1].hist(attention_weights[:, 0], bins=20, alpha=0.7, label='Genus Attention', color='blue')
    axes[1,1].hist(attention_weights[:, 1], bins=20, alpha=0.7, label='Metabolite Attention', color='red')
    axes[1,1].axvline(x=avg_genus_attention, color='blue', linestyle='--', 
                     label=f'Genus Avg: {avg_genus_attention:.3f}')
    axes[1,1].axvline(x=avg_metab_attention, color='red', linestyle='--', 
                     label=f'Metab Avg: {avg_metab_attention:.3f}')
    axes[1,1].set_xlabel('Attention Weight')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Attention Weight Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. 预测vs真实标签
    jitter = np.random.normal(0, 0.05, len(y_true))
    scatter = axes[1,2].scatter(y_true + jitter, y_prob, alpha=0.6, c=y_true, cmap='RdYlBu_r', s=30)
    axes[1,2].axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
    axes[1,2].set_xlabel('True Label')
    axes[1,2].set_ylabel('Predicted Probability')
    axes[1,2].set_title('Predicted Probability vs True Label')
    axes[1,2].set_xticks([0, 1])
    axes[1,2].set_xticklabels(['Non-allergenic (0)', 'Allergenic (1)'])
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('allergy_classification_results_324.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🧬 多模态过敏性疾病分类器 - 324数据集")
    print("=" * 70)
    print("📋 数据说明:")
    print("   • 1 (IgE): 引起过敏反应的样本")
    print("   • 0 (N group): 非过敏性样本（正常对照组）")
    print("   • 使用所有特征: 567个genus + 324个metabolite = 891个特征")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\\n💻 使用设备: {device}")
    
    # 加载数据
    print("\\n📊 加载数据...")
    try:
        # X_train = pd.read_excel('324_X_train_source115_genus_metabolite.xlsx', index_col=0)
        # y_train = pd.read_excel('324_y_train_source115_genus_metabolite.xlsx', index_col=0)
        # X_test = pd.read_excel('324_X_test_source203_genus_metabolite.xlsx', index_col=0)
        # y_test = pd.read_excel('324_y_test_source203_genus_metabolite.xlsx', index_col=0)

        X_train = pd.read_excel('324_X_test_source203_genus_metabolite.xlsx', index_col=0)
        y_train = pd.read_excel('324_y_test_source203_genus_metabolite.xlsx', index_col=0)
        X_test = pd.read_excel('324_X_train_source115_genus_metabolite.xlsx', index_col=0)
        y_test = pd.read_excel('324_y_train_source115_genus_metabolite.xlsx', index_col=0)
        
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
    
    print(f"\\n📈 类别分布:")
    train_counts = np.bincount(y_train_values)
    test_counts = np.bincount(y_test_values)
    print(f"   训练集: {train_counts[0]}个非过敏样本(N), {train_counts[1]}个过敏样本(IgE)")
    print(f"   测试集: {test_counts[0]}个非过敏样本(N), {test_counts[1]}个过敏样本(IgE)")
    
    # 分离多模态特征
    genus_features = [col for col in X_train.columns if 'genus' in col.lower()]
    metabolite_features = [col for col in X_train.columns if 'metabolite' in col.lower()]
    
    X_train_genus = X_train[genus_features].values
    X_train_metab = X_train[metabolite_features].values
    X_test_genus = X_test[genus_features].values
    X_test_metab = X_test[metabolite_features].values
    
    print(f"\\n🔬 特征分析:")
    print(f"   Genus特征 (微生物组): {len(genus_features)}个")
    print(f"   Metabolite特征 (代谢物): {len(metabolite_features)}个")
    print(f"   总特征数: {len(genus_features) + len(metabolite_features)}个")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\\n💻 使用设备: {device}")
    
    # 数据标准化
    print("\\n📊 数据标准化...")
    scaler_genus = StandardScaler()
    scaler_metab = StandardScaler()
    
    X_train_genus_scaled = scaler_genus.fit_transform(X_train_genus)
    X_train_metab_scaled = scaler_metab.fit_transform(X_train_metab)
    
    X_test_genus_scaled = scaler_genus.transform(X_test_genus)
    X_test_metab_scaled = scaler_metab.transform(X_test_metab)
    
    print(f"✅ 标准化完成:")
    print(f"   Genus特征维度: {X_train_genus_scaled.shape[1]}")
    print(f"   Metabolite特征维度: {X_train_metab_scaled.shape[1]}")
    
    # 创建模型
    print("\\n🤖 创建多模态注意力分类器...")
    model = EnhancedMultimodalClassifier(
        micro_input_dim=X_train_genus_scaled.shape[1],
        metab_input_dim=X_train_metab_scaled.shape[1],
        hidden_dim=128,  # 由于特征多，使用更大的隐藏层
        dropout_rate=0.4
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 处理类别不平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_values), y=y_train_values)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"\\n⚖️ 类别权重 (处理不平衡): 非过敏={class_weight_dict[0]:.3f}, 过敏={class_weight_dict[1]:.3f}")
    
    # 加权采样器
    sample_weights = [class_weight_dict[label] for label in y_train_values]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # 数据加载器
    train_loader = create_data_loader(
        X_train_genus_scaled, X_train_metab_scaled, y_train_values,
        batch_size=16, sampler=sampler  # 较小的batch size适合高维特征
    )
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.BCELoss(weight=torch.FloatTensor([class_weights[1]]).to(device))
    
    # 训练循环
    print("\\n🚀 开始训练...")
    print("=" * 70)
    train_losses = []
    val_aucs = []
    best_auc = 0
    patience = 15
    no_improve_count = 0
    
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
                torch.save(model.state_dict(), 'best_allergy_model_324.pth')
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        model.train()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Val_AUC={val_auc:.4f}, Best_AUC={best_auc:.4f}")
        
        # 早停
        if no_improve_count >= patience:
            print(f"  早停于第{epoch+1}轮 (验证AUC连续{patience}轮未提升)")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_allergy_model_324.pth'))
    
    # 最终评估 - 分批处理
    print("\\n" + "="*70)
    print("🎯 最终评估结果 - 过敏性疾病分类")
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
    
    print("\\n📊 详细分类报告:")
    print(classification_report(y_test_values, test_pred_binary, 
                              target_names=['Non-allergenic (N)', 'Allergenic (IgE)'],
                              digits=4))
    
    # 注意力分析
    avg_genus_attention = attention_weights[:, 0].mean()
    avg_metab_attention = attention_weights[:, 1].mean()
    print(f"\\n🎯 注意力权重分析:")
    print(f"   平均Genus注意力权重: {avg_genus_attention:.4f}")
    print(f"   平均Metabolite注意力权重: {avg_metab_attention:.4f}")
    
    if avg_genus_attention > avg_metab_attention:
        print(f"   → 模型更关注微生物组特征 (差异: {avg_genus_attention - avg_metab_attention:.4f})")
    else:
        print(f"   → 模型更关注代谢物特征 (差异: {avg_metab_attention - avg_genus_attention:.4f})")
    
    # 可视化
    print("\\n📈 生成可视化图表...")
    plot_training_curves(train_losses, val_aucs)
    plot_results(y_test_values, test_pred_binary, test_pred_prob, attention_weights)
    
    # 保存结果
    results_df = pd.DataFrame({
        'sample_id': X_test.index,
        'true_label': y_test_values,
        'true_label_name': ['Non-allergenic' if x == 0 else 'Allergenic' for x in y_test_values],
        'predicted_prob': test_pred_prob,
        'predicted_label': test_pred_binary,
        'predicted_label_name': ['Non-allergenic' if x == 0 else 'Allergenic' for x in test_pred_binary],
        'genus_attention': attention_weights[:, 0],
        'metabolite_attention': attention_weights[:, 1],
        'correct_prediction': (test_pred_binary == y_test_values).astype(int)
    })
    results_df.to_csv('allergy_prediction_results_324.csv', index=False)
    
    # 统计错误分类的样本
    false_positives = results_df[(results_df['true_label'] == 0) & (results_df['predicted_label'] == 1)]
    false_negatives = results_df[(results_df['true_label'] == 1) & (results_df['predicted_label'] == 0)]
    
    print(f"\\n📋 错误分类分析:")
    print(f"   假阳性 (误判为过敏): {len(false_positives)}个样本")
    print(f"   假阴性 (漏判过敏): {len(false_negatives)}个样本")
    
    print("\\n✅ 训练完成!")
    print("📁 输出文件:")
    print("   • best_allergy_model_324.pth - 最佳模型权重")
    print("   • training_curves_324_allergy.png - 训练曲线")
    print("   • allergy_classification_results_324.png - 结果分析图")
    print("   • allergy_prediction_results_324.csv - 详细预测结果")

if __name__ == "__main__":
    main()
