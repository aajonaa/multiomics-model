#!/usr/bin/env python3
"""
训练324数据集的多模态分类器 - 10折交叉验证
使用source 115数据集 (108个样本) 进行10折交叉验证: 
- 只使用 324_X_train_source115_genus_metabolite.xlsx (108个样本)
- 10折交叉验证：每折约10-11个样本作为测试集，其余作为训练集
- 不使用source 203数据

数据分割策略:
- Source 115 (108个样本) -> 10折交叉验证 (每折: ~97个训练 + ~11个测试)
- 每折内部再分: 训练集 (~78个) + 验证集 (~19个) + 测试集 (~11个)

标签含义:
- 1 (IgE): 引起过敏反应的样本
- 0 (N group): 非过敏性样本（正常对照组）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
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

def create_data_loader(X_micro, X_metab, y, batch_size=16, shuffle=True, sampler=None):
    """创建PyTorch数据加载器"""
    X_micro_tensor = torch.FloatTensor(X_micro)
    X_metab_tensor = torch.FloatTensor(X_metab)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    dataset = TensorDataset(X_micro_tensor, X_metab_tensor, y_tensor)
    
    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for x_micro, x_metab, labels in data_loader:
            x_micro, x_metab, labels = x_micro.to(device), x_metab.to(device), labels.to(device)
            outputs = model(x_micro, x_metab)
            predictions = outputs['classification'].cpu().numpy().flatten()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy().flatten())
    
    if len(all_predictions) == 0:
        return {'auc': 0.0, 'accuracy': 0.0, 'predictions': np.array([]), 'labels': np.array([])}
    
    auc = roc_auc_score(all_labels, all_predictions)
    binary_pred = (np.array(all_predictions) > 0.5).astype(int)
    accuracy = (binary_pred == np.array(all_labels)).mean()
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }

def train_single_fold(X_train_genus, X_train_metab, y_train, X_val_genus, X_val_metab, y_val, 
                     X_test_genus, X_test_metab, y_test, device, fold_num):
    """训练单个折叠"""
    
    # 创建模型
    model = EnhancedMultimodalClassifier(
        micro_input_dim=X_train_genus.shape[1],
        metab_input_dim=X_train_metab.shape[1],
        hidden_dim=64,  # 减小模型以适应小数据集
        dropout_rate=0.3
    ).to(device)
    
    # 处理类别不平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # 创建数据加载器
    sample_weights = [class_weight_dict[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = create_data_loader(X_train_genus, X_train_metab, y_train, 
                                    batch_size=8, sampler=sampler)  # 小batch size
    val_loader = create_data_loader(X_val_genus, X_val_metab, y_val, 
                                  batch_size=8, shuffle=False)
    test_loader = create_data_loader(X_test_genus, X_test_metab, y_test, 
                                   batch_size=8, shuffle=False)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.BCELoss(weight=torch.FloatTensor([class_weights[1]]).to(device))
    
    # 训练循环
    best_val_auc = 0
    patience = 10
    no_improve_count = 0
    
    for epoch in range(50):  # 减少训练轮数
        # 训练阶段
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for x_genus, x_metab, labels in train_loader:
            x_genus, x_metab, labels = x_genus.to(device), x_metab.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_genus, x_metab)
            
            loss = criterion(outputs['classification'], labels)
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            total_loss = loss + 0.01 * l2_reg  # 增加正则化
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # 验证阶段
        if len(y_val) > 0:
            val_results = evaluate_model(model, val_loader, device)
            val_auc = val_results['auc']
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                break
        else:
            # 如果没有验证集，保存最后的模型
            best_model_state = model.state_dict().copy()
    
    # 加载最佳模型并评估测试集
    model.load_state_dict(best_model_state)
    test_results = evaluate_model(model, test_loader, device)
    
    # 获取注意力权重
    model.eval()
    attention_weights_list = []
    with torch.no_grad():
        for x_genus, x_metab, _ in test_loader:
            x_genus, x_metab = x_genus.to(device), x_metab.to(device)
            outputs = model(x_genus, x_metab)
            attention_weights_list.append(outputs['attention_weights'].cpu().numpy())
    
    if attention_weights_list:
        attention_weights = np.vstack(attention_weights_list)
    else:
        attention_weights = np.array([[0.5, 0.5]])  # 默认权重
    
    return {
        'test_auc': test_results['auc'],
        'test_accuracy': test_results['accuracy'],
        'test_predictions': test_results['predictions'],
        'test_labels': test_results['labels'],
        'attention_weights': attention_weights,
        'best_val_auc': best_val_auc
    }

def plot_cv_results(cv_results):
    """绘制交叉验证结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 每折AUC分布
    fold_aucs = [result['test_auc'] for result in cv_results]
    axes[0,0].bar(range(1, len(fold_aucs)+1), fold_aucs, alpha=0.7, color='steelblue')
    axes[0,0].axhline(y=np.mean(fold_aucs), color='red', linestyle='--', 
                     label=f'Mean AUC: {np.mean(fold_aucs):.3f}')
    axes[0,0].set_xlabel('Fold')
    axes[0,0].set_ylabel('Test AUC')
    axes[0,0].set_title('10-Fold Cross-Validation AUC Results')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. AUC分布直方图
    axes[0,1].hist(fold_aucs, bins=8, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0,1].axvline(x=np.mean(fold_aucs), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(fold_aucs):.3f}')
    axes[0,1].axvline(x=np.median(fold_aucs), color='green', linestyle='--', 
                     label=f'Median: {np.median(fold_aucs):.3f}')
    axes[0,1].set_xlabel('AUC Score')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Distribution of AUC Scores Across Folds')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 注意力权重分析
    all_attention = np.vstack([result['attention_weights'] for result in cv_results if len(result['attention_weights']) > 0])
    avg_genus_attention = all_attention[:, 0].mean()
    avg_metab_attention = all_attention[:, 1].mean()
    
    modalities = ['Genus\\n(Microbiome)', 'Metabolite']
    avg_weights = [avg_genus_attention, avg_metab_attention]
    colors = ['lightcoral', 'lightblue']
    
    bars = axes[1,0].bar(modalities, avg_weights, color=colors, alpha=0.8)
    axes[1,0].set_ylabel('Average Attention Weight')
    axes[1,0].set_title('Average Attention Weights Across All Folds')
    axes[1,0].set_ylim(0, 1)
    axes[1,0].grid(axis='y', alpha=0.3)
    
    for bar, weight in zip(bars, avg_weights):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                      f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 统计汇总
    axes[1,1].axis('off')
    stats_text = f"""
10-Fold Cross-Validation Results Summary

AUC Statistics:
• Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}
• Median AUC: {np.median(fold_aucs):.4f}
• Min AUC: {np.min(fold_aucs):.4f}
• Max AUC: {np.max(fold_aucs):.4f}
• Range: {np.max(fold_aucs) - np.min(fold_aucs):.4f}

Model Attention:
• Genus (Microbiome): {avg_genus_attention:.1%}
• Metabolite: {avg_metab_attention:.1%}

Dataset:
• Total Samples: 108
• Each Fold Test Size: ~11 samples
• Features: 891 (567 genus + 324 metabolite)
"""
    axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes, 
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('10_fold_cv_results_source115.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🧬 多模态过敏性疾病分类器 - 10折交叉验证")
    print("=" * 80)
    print("📋 验证策略:")
    print("   • 只使用 Source 115 数据集 (108个样本)")
    print("   • 10折交叉验证: 每折~11个测试样本, ~97个训练+验证样本")
    print("   • 每折内部: ~78个训练 + ~19个验证 + ~11个测试")
    print("   • 标签: 1 (IgE过敏), 0 (N正常)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\\n💻 使用设备: {device}")
    
    # 加载数据 - 只使用source 115
    print("\\n📊 加载数据...")
    try:
        X_data = pd.read_excel('324_X_train_source115_genus_metabolite.xlsx', index_col=0)
        y_data = pd.read_excel('324_y_train_source115_genus_metabolite.xlsx', index_col=0)
        
        print(f"✅ 数据加载成功")
        print(f"   总样本数: {X_data.shape[0]}个")
        print(f"   特征数: {X_data.shape[1]}个")
        
    except FileNotFoundError as e:
        print(f"❌ 数据文件未找到: {e}")
        return
    
    # 清理列名
    X_data = clean_column_names(X_data)
    y_values = y_data.iloc[:, 0].values
    
    print(f"\\n📈 数据类别分布:")
    class_counts = np.bincount(y_values)
    print(f"   非过敏样本 (N): {class_counts[0]}个 ({class_counts[0]/len(y_values)*100:.1f}%)")
    print(f"   过敏样本 (IgE): {class_counts[1]}个 ({class_counts[1]/len(y_values)*100:.1f}%)")
    
    # 分离多模态特征
    genus_features = [col for col in X_data.columns if 'genus' in col.lower()]
    metabolite_features = [col for col in X_data.columns if 'metabolite' in col.lower()]
    
    X_genus = X_data[genus_features].values
    X_metab = X_data[metabolite_features].values
    
    print(f"\\n🔬 特征分析:")
    print(f"   Genus特征: {len(genus_features)}个")
    print(f"   Metabolite特征: {len(metabolite_features)}个")
    print(f"   总特征数: {len(genus_features) + len(metabolite_features)}个")
    
    # 10折交叉验证
    print("\\n🔄 开始10折交叉验证...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X_genus, y_values)):
        print(f"\\n--- 第 {fold+1}/10 折 ---")
        
        # 分割数据
        X_train_val_genus, X_test_fold_genus = X_genus[train_val_idx], X_genus[test_idx]
        X_train_val_metab, X_test_fold_metab = X_metab[train_val_idx], X_metab[test_idx]
        y_train_val, y_test_fold = y_values[train_val_idx], y_values[test_idx]
        
        # 进一步分割训练验证集
        if len(y_train_val) > 10:  # 只有足够样本时才分割验证集
            X_train_genus, X_val_genus, X_train_metab, X_val_metab, y_train, y_val = train_test_split(
                X_train_val_genus, X_train_val_metab, y_train_val, 
                test_size=0.2, random_state=42, stratify=y_train_val
            )
        else:
            # 样本太少，不分验证集
            X_train_genus, X_val_genus = X_train_val_genus, np.array([]).reshape(0, X_train_val_genus.shape[1])
            X_train_metab, X_val_metab = X_train_val_metab, np.array([]).reshape(0, X_train_val_metab.shape[1])
            y_train, y_val = y_train_val, np.array([])
        
        print(f"   训练集: {len(y_train)}个样本")
        print(f"   验证集: {len(y_val)}个样本")
        print(f"   测试集: {len(y_test_fold)}个样本")
        
        # 数据标准化
        scaler_genus = StandardScaler()
        scaler_metab = StandardScaler()
        
        X_train_genus_scaled = scaler_genus.fit_transform(X_train_genus)
        X_train_metab_scaled = scaler_metab.fit_transform(X_train_metab)
        
        if len(y_val) > 0:
            X_val_genus_scaled = scaler_genus.transform(X_val_genus)
            X_val_metab_scaled = scaler_metab.transform(X_val_metab)
        else:
            X_val_genus_scaled = np.array([]).reshape(0, X_train_genus_scaled.shape[1])
            X_val_metab_scaled = np.array([]).reshape(0, X_train_metab_scaled.shape[1])
        
        X_test_fold_genus_scaled = scaler_genus.transform(X_test_fold_genus)
        X_test_fold_metab_scaled = scaler_metab.transform(X_test_fold_metab)
        
        # 训练模型
        fold_result = train_single_fold(
            X_train_genus_scaled, X_train_metab_scaled, y_train,
            X_val_genus_scaled, X_val_metab_scaled, y_val,
            X_test_fold_genus_scaled, X_test_fold_metab_scaled, y_test_fold,
            device, fold+1
        )
        
        cv_results.append(fold_result)
        print(f"   测试AUC: {fold_result['test_auc']:.4f}")
        print(f"   测试准确率: {fold_result['test_accuracy']:.4f}")
    
    # 汇总结果
    print("\\n" + "="*80)
    print("🎯 10折交叉验证总结")
    print("="*80)
    
    fold_aucs = [result['test_auc'] for result in cv_results]
    fold_accs = [result['test_accuracy'] for result in cv_results]
    
    print(f"🔥 平均测试AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"🔥 平均测试准确率: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"🔥 AUC范围: [{np.min(fold_aucs):.4f}, {np.max(fold_aucs):.4f}]")
    print(f"🔥 中位数AUC: {np.median(fold_aucs):.4f}")
    
    # 显示每折结果
    print(f"\\n📊 各折详细结果:")
    for i, result in enumerate(cv_results):
        print(f"   第{i+1:2d}折: AUC={result['test_auc']:.4f}, Acc={result['test_accuracy']:.4f}")
    
    # 注意力权重分析
    all_attention = np.vstack([result['attention_weights'] for result in cv_results if len(result['attention_weights']) > 0])
    avg_genus_attention = all_attention[:, 0].mean()
    avg_metab_attention = all_attention[:, 1].mean()
    
    print(f"\\n🎯 平均注意力权重:")
    print(f"   Genus (微生物组): {avg_genus_attention:.4f} ({avg_genus_attention:.1%})")
    print(f"   Metabolite (代谢物): {avg_metab_attention:.4f} ({avg_metab_attention:.1%})")
    
    # 可视化结果
    print("\\n📈 生成可视化图表...")
    plot_cv_results(cv_results)
    
    # 保存结果
    results_summary = pd.DataFrame({
        'fold': range(1, 11),
        'test_auc': fold_aucs,
        'test_accuracy': fold_accs
    })
    results_summary.to_csv('10_fold_cv_results_summary.csv', index=False)
    
    # 保存统计汇总
    stats_summary = pd.DataFrame({
        'metric': ['mean_auc', 'std_auc', 'median_auc', 'min_auc', 'max_auc', 
                  'mean_accuracy', 'std_accuracy', 'avg_genus_attention', 'avg_metabolite_attention'],
        'value': [np.mean(fold_aucs), np.std(fold_aucs), np.median(fold_aucs), 
                 np.min(fold_aucs), np.max(fold_aucs), np.mean(fold_accs), np.std(fold_accs),
                 avg_genus_attention, avg_metab_attention]
    })
    stats_summary.to_csv('cv_statistics_summary.csv', index=False)
    
    print("\\n✅ 10折交叉验证完成!")
    print("📁 输出文件:")
    print("   • 10_fold_cv_results_source115.png - 交叉验证结果可视化")
    print("   • 10_fold_cv_results_summary.csv - 各折详细结果")
    print("   • cv_statistics_summary.csv - 统计汇总")

if __name__ == "__main__":
    main()
