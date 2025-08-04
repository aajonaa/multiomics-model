import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def analyze_metabolite_data_structure():
    """正确分析代谢物数据结构"""
    
    # 文件路径
    file_115 = r"代谢数据\115例代谢\metabolite_115.xlsx"
    file_203 = r"代谢数据\203例代谢\metabolite_203.xlsx"
    
    print("=== 正确分析代谢物数据结构 ===")
    
    results = {}
    
    for file_path, name in [(file_115, "115例"), (file_203, "203例")]:
        print(f"\n{'='*50}")
        print(f"分析 {name} 数据")
        print(f"{'='*50}")
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        try:
            # 读取差异表达矩阵sheet
            df = pd.read_excel(file_path, sheet_name='差异表达矩阵')
            print(f"数据形状: {df.shape}")
            print(f"代谢物数量: {df.shape[0]}")
            print(f"总列数: {df.shape[1]}")
            
            # 识别元数据列和样本列
            metadata_cols = []
            sample_cols = []
            
            for col in df.columns:
                # 如果列名包含样本编号模式，认为是样本列
                if any(pattern in str(col) for pattern in ['IgE-', 'N-']) and \
                   not any(word in str(col).lower() for word in ['class', 'annotation', 'name', 'description']):
                    sample_cols.append(col)
                else:
                    metadata_cols.append(col)
            
            print(f"元数据列数: {len(metadata_cols)}")
            print(f"样本列数: {len(sample_cols)}")
            
            print(f"\n元数据列: {metadata_cols[:10]}{'...' if len(metadata_cols) > 10 else ''}")
            print(f"样本列示例: {sample_cols[:10]}{'...' if len(sample_cols) > 10 else ''}")
            
            # 检查数据质量
            print(f"\n=== {name} 数据质量检查 ===")
            
            # 样本数据的基本统计
            if len(sample_cols) > 0:
                sample_data = df[sample_cols]
                print(f"样本数据基本统计:")
                print(f"  - 平均值范围: {sample_data.mean().min():.2f} ~ {sample_data.mean().max():.2f}")
                print(f"  - 缺失值数量: {sample_data.isnull().sum().sum()}")
                print(f"  - 零值数量: {(sample_data == 0).sum().sum()}")
            
            results[name] = {
                'df': df,
                'metadata_cols': metadata_cols,
                'sample_cols': sample_cols,
                'shape': df.shape
            }
            
        except Exception as e:
            print(f"读取文件时出错: {e}")
            results[name] = None
    
    return results

def find_common_metabolites(results):
    """找到两个数据集中共同的代谢物"""
    
    if '115例' not in results or '203例' not in results or \
       results['115例'] is None or results['203例'] is None:
        print("无法比较代谢物，数据缺失")
        return None
    
    print(f"\n{'='*50}")
    print("寻找共同代谢物")
    print(f"{'='*50}")
    
    df_115 = results['115例']['df']
    df_203 = results['203例']['df']
    
    # 使用ID列来匹配代谢物
    metabolites_115 = set(df_115['Metabolites'])
    metabolites_203 = set(df_203['Metabolites'])
    
    common_metabolites = metabolites_115.intersection(metabolites_203)
    only_115 = metabolites_115 - metabolites_203
    only_203 = metabolites_203 - metabolites_115
    
    print(f"115例数据中的代谢物数量: {len(metabolites_115)}")
    print(f"203例数据中的代谢物数量: {len(metabolites_203)}")
    print(f"共同代谢物数量: {len(common_metabolites)}")
    print(f"仅在115例中的代谢物数量: {len(only_115)}")
    print(f"仅在203例中的代谢物数量: {len(only_203)}")
    
    if len(common_metabolites) > 0:
        print(f"\n前10个共同代谢物ID: {list(common_metabolites)[:10]}")
    
    return {
        'common_metabolites': common_metabolites,
        'only_115': only_115,
        'only_203': only_203
    }

def merge_metabolite_datasets(results, common_metabolites_info):
    """基于共同代谢物合并数据集"""
    
    if results['115例'] is None or results['203例'] is None:
        print("无法合并数据，数据缺失")
        return None
    
    if len(common_metabolites_info['common_metabolites']) == 0:
        print("没有共同代谢物，无法合并")
        return None
    
    print(f"\n{'='*50}")
    print("合并数据集")
    print(f"{'='*50}")
    
    df_115 = results['115例']['df']
    df_203 = results['203例']['df']
    sample_cols_115 = results['115例']['sample_cols']
    sample_cols_203 = results['203例']['sample_cols']
    metadata_cols_115 = results['115例']['metadata_cols']
    
    common_metabolites = common_metabolites_info['common_metabolites']
    
    print(f"使用 {len(common_metabolites)} 个共同代谢物进行合并")
    print(f"115例数据样本数: {len(sample_cols_115)}")
    print(f"203例数据样本数: {len(sample_cols_203)}")
    
    # 筛选共同代谢物 - 使用Metabolites列而不是ID列
    df_115_common = df_115[df_115['Metabolites'].isin(common_metabolites)].copy()
    df_203_common = df_203[df_203['Metabolites'].isin(common_metabolites)].copy()
    
    print(f"筛选后115例数据形状: {df_115_common.shape}")
    print(f"筛选后203例数据形状: {df_203_common.shape}")
    
    # 按Metabolites排序确保一致性
    df_115_common = df_115_common.sort_values('Metabolites').reset_index(drop=True)
    df_203_common = df_203_common.sort_values('Metabolites').reset_index(drop=True)
    
    # 合并元数据（使用115例的元数据作为基础）
    merged_metadata = df_115_common[metadata_cols_115].copy()
    
    # 合并样本数据
    merged_samples = pd.concat([
        df_115_common[sample_cols_115],
        df_203_common[sample_cols_203]
    ], axis=1)
    
    # 合并完整数据
    merged_df = pd.concat([merged_metadata, merged_samples], axis=1)
    
    total_samples = len(sample_cols_115) + len(sample_cols_203)
    
    print(f"\n合并结果:")
    print(f"  - 共同代谢物数量: {len(common_metabolites)}")
    print(f"  - 总样本数: {total_samples} (115例: {len(sample_cols_115)}, 203例: {len(sample_cols_203)})")
    print(f"  - 合并后数据形状: {merged_df.shape}")
    
    return {
        'merged_df': merged_df,
        'sample_cols_115': sample_cols_115,
        'sample_cols_203': sample_cols_203,
        'total_samples': total_samples,
        'common_metabolites_count': len(common_metabolites)
    }

def save_common_metabolites_info(results, common_metabolites_info, output_file="common_metabolites_detailed_info.xlsx"):
    """保存共同代谢物的详细信息到Excel文件"""
    
    if not common_metabolites_info or len(common_metabolites_info['common_metabolites']) == 0:
        print("没有共同代谢物信息可保存")
        return None
    
    print(f"\n{'='*50}")
    print("生成共同代谢物详细信息文件")
    print(f"{'='*50}")
    
    df_115 = results['115例']['df']
    df_203 = results['203例']['df']
    common_metabolites = common_metabolites_info['common_metabolites']
    
    # 获取115例中的共同代谢物信息
    common_115 = df_115[df_115['Metabolites'].isin(common_metabolites)].copy()
    common_203 = df_203[df_203['Metabolites'].isin(common_metabolites)].copy()
    
    # 按Metabolites排序
    common_115 = common_115.sort_values('Metabolites').reset_index(drop=True)
    common_203 = common_203.sort_values('Metabolites').reset_index(drop=True)
    
    # 选择关键的元数据列进行保存
    key_columns = ['ID', 'Metabolites', 'Metabolites_cn', 'm/z', 'Retention time (min)', 'Ion mode']
    
    # 确保列存在
    available_cols_115 = [col for col in key_columns if col in common_115.columns]
    available_cols_203 = [col for col in key_columns if col in common_203.columns]
    
    # 准备115例数据
    info_115 = common_115[available_cols_115].copy()
    info_115['数据来源'] = '115例'
    
    # 准备203例数据
    info_203 = common_203[available_cols_203].copy()
    info_203['数据来源'] = '203例'
    
    # 创建Excel写入器
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Sheet 1: 共同代谢物概览
        overview_data = {
            '统计项目': [
                '115例代谢物总数',
                '203例代谢物总数', 
                '共同代谢物数量',
                '仅在115例中的代谢物',
                '仅在203例中的代谢物',
                '共同代谢物占115例比例(%)',
                '共同代谢物占203例比例(%)'
            ],
            '数值': [
                len(set(df_115['Metabolites'])),
                len(set(df_203['Metabolites'])),
                len(common_metabolites),
                len(common_metabolites_info['only_115']),
                len(common_metabolites_info['only_203']),
                round(len(common_metabolites) / len(set(df_115['Metabolites'])) * 100, 2),
                round(len(common_metabolites) / len(set(df_203['Metabolites'])) * 100, 2)
            ]
        }
        overview_df = pd.DataFrame(overview_data)
        overview_df.to_excel(writer, sheet_name='共同代谢物概览', index=False)
        
        # Sheet 2: 115例共同代谢物详情
        info_115.to_excel(writer, sheet_name='115例共同代谢物', index=False)
        
        # Sheet 3: 203例共同代谢物详情
        info_203.to_excel(writer, sheet_name='203例共同代谢物', index=False)
        
        # Sheet 4: 共同代谢物名称列表
        # 创建代谢物名称和中文名称的映射
        metabolite_cn_mapping = {}
        
        # 从115例数据中获取中文名称
        for _, row in common_115.iterrows():
            metabolite_name = row['Metabolites']
            cn_name = row.get('Metabolites_cn', '')
            if pd.notna(cn_name) and cn_name.strip():
                metabolite_cn_mapping[metabolite_name] = cn_name
        
        # 从203例数据中补充中文名称（如果115例中没有的话）
        for _, row in common_203.iterrows():
            metabolite_name = row['Metabolites']
            cn_name = row.get('Metabolites_cn', '')
            if metabolite_name not in metabolite_cn_mapping and pd.notna(cn_name) and cn_name.strip():
                metabolite_cn_mapping[metabolite_name] = cn_name
        
        # 创建包含中文名称的代谢物列表
        metabolites_with_cn = []
        for i, metabolite in enumerate(sorted(list(common_metabolites))):
            cn_name = metabolite_cn_mapping.get(metabolite, '无中文名称')
            metabolites_with_cn.append({
                '序号': i + 1,
                '代谢物名称': metabolite,
                '代谢物名称（中文）': cn_name
            })
        
        metabolites_list = pd.DataFrame(metabolites_with_cn)
        metabolites_list.to_excel(writer, sheet_name='共同代谢物列表', index=False)
        
        # Sheet 5: 仅在115例中的代谢物
        if len(common_metabolites_info['only_115']) > 0:
            only_115_df = pd.DataFrame({
                '序号': range(1, len(common_metabolites_info['only_115']) + 1),
                '仅在115例中的代谢物': sorted(list(common_metabolites_info['only_115']))
            })
            only_115_df.to_excel(writer, sheet_name='仅115例代谢物', index=False)
        
        # Sheet 6: 仅在203例中的代谢物
        if len(common_metabolites_info['only_203']) > 0:
            only_203_df = pd.DataFrame({
                '序号': range(1, len(common_metabolites_info['only_203']) + 1),
                '仅在203例中的代谢物': sorted(list(common_metabolites_info['only_203']))
            })
            only_203_df.to_excel(writer, sheet_name='仅203例代谢物', index=False)
    
    print(f"共同代谢物详细信息已保存到: {output_file}")
    print(f"包含以下工作表:")
    print(f"  1. 共同代谢物概览 - 统计信息")
    print(f"  2. 115例共同代谢物 - 115例中共同代谢物的详细信息")
    print(f"  3. 203例共同代谢物 - 203例中共同代谢物的详细信息")
    print(f"  4. 共同代谢物列表 - 所有共同代谢物名称")
    print(f"  5. 仅115例代谢物 - 只在115例中存在的代谢物")
    print(f"  6. 仅203例代谢物 - 只在203例中存在的代谢物")
    
    return output_file

def save_merged_data(merge_result, output_prefix="merged_metabolite"):
    """保存合并后的数据"""
    
    if merge_result is None:
        print("没有数据可保存")
        return
    
    merged_df = merge_result['merged_df']
    
    # 保存完整合并数据
    full_output = f"{output_prefix}_full_data.xlsx"
    merged_df.to_excel(full_output, index=False)
    print(f"\n完整合并数据已保存到: {full_output}")
    
    # 创建样本-特征矩阵（转置，用于机器学习）
    sample_cols = merge_result['sample_cols_115'] + merge_result['sample_cols_203']
    
    # 提取样本数据并转置
    sample_matrix = merged_df[sample_cols].T  # 转置：样本作为行，代谢物作为列
    sample_matrix.columns = merged_df['Metabolites']  # 使用代谢物作为列名
    
    # 添加样本信息
    sample_matrix = sample_matrix.reset_index()  # 重置索引
    sample_matrix.rename(columns={'index': 'sample_id'}, inplace=True)
    
    print(f"Debug: sample_matrix shape after transpose and reset: {sample_matrix.shape}")
    print(f"Debug: expected sample count: {len(merge_result['sample_cols_115'])} + {len(merge_result['sample_cols_203'])} = {len(merge_result['sample_cols_115']) + len(merge_result['sample_cols_203'])}")
    print(f"Debug: actual sample_matrix rows: {len(sample_matrix)}")
    
    # 创建数据来源标识，确保长度匹配
    if len(sample_matrix) == len(merge_result['sample_cols_115']) + len(merge_result['sample_cols_203']):
        data_source = ['115_samples'] * len(merge_result['sample_cols_115']) + \
                      ['203_samples'] * len(merge_result['sample_cols_203'])
    else:
        # 如果长度不匹配，根据实际样本ID来判断来源
        data_source = []
        for sample_id in sample_matrix['sample_id']:
            if sample_id in merge_result['sample_cols_115']:
                data_source.append('115_samples')
            elif sample_id in merge_result['sample_cols_203']:
                data_source.append('203_samples')
            else:
                data_source.append('unknown')
    
    sample_matrix['data_source'] = data_source
    
    # 重新排列列，将样本信息放在前面
    cols = ['sample_id', 'data_source'] + [col for col in sample_matrix.columns if col not in ['sample_id', 'data_source']]
    sample_matrix = sample_matrix[cols]
    
    ml_output = f"{output_prefix}_ml_format.xlsx"
    sample_matrix.to_excel(ml_output, index=False)
    print(f"机器学习格式数据已保存到: {ml_output}")
    
    # 打印统计信息
    print(f"\n=== 合并数据统计 ===")
    print(f"代谢物数量: {merge_result['common_metabolites_count']}")
    print(f"总样本数: {merge_result['total_samples']}")
    print(f"数据来源分布:")
    print(f"  - 115例样本: {len(merge_result['sample_cols_115'])}")
    print(f"  - 203例样本: {len(merge_result['sample_cols_203'])}")
    
    # 检查数据质量
    numeric_data = sample_matrix.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 0:
        print(f"\n数据质量检查:")
        print(f"  - 缺失值总数: {numeric_data.isnull().sum().sum()}")
        print(f"  - 零值总数: {(numeric_data == 0).sum().sum()}")
        print(f"  - 数值范围: {numeric_data.min().min():.2f} ~ {numeric_data.max().max():.2f}")
    
    return {
        'full_data_file': full_output,
        'ml_format_file': ml_output,
        'sample_matrix': sample_matrix
    }

def create_visualization(results, merge_result):
    """创建数据比较和合并结果的可视化"""
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 代谢物数量比较
        metabolite_counts = [
            results['115例']['shape'][0] if results['115例'] else 0,
            results['203例']['shape'][0] if results['203例'] else 0,
            merge_result['common_metabolites_count'] if merge_result else 0
        ]
        labels = ['115例代谢物', '203例代谢物', '共同代谢物']
        bars1 = axes[0, 0].bar(labels, metabolite_counts, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('代谢物数量比较')
        axes[0, 0].set_ylabel('代谢物数量')
        
        # 添加数值标签
        for i, v in enumerate(metabolite_counts):
            axes[0, 0].text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        # 2. 样本数量比较
        if merge_result:
            sample_counts = [
                len(merge_result['sample_cols_115']),
                len(merge_result['sample_cols_203']),
                merge_result['total_samples']
            ]
            sample_labels = ['115例样本', '203例样本', '合并总样本']
            bars2 = axes[0, 1].bar(sample_labels, sample_counts, color=['orange', 'purple', 'green'])
            axes[0, 1].set_title('样本数量比较')
            axes[0, 1].set_ylabel('样本数量')
            
            # 添加数值标签
            for i, v in enumerate(sample_counts):
                axes[0, 1].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # 3. 数据来源分布饼图
        if merge_result:
            source_counts = [len(merge_result['sample_cols_115']), len(merge_result['sample_cols_203'])]
            source_labels = ['115例样本', '203例样本']
            axes[0, 2].pie(source_counts, labels=source_labels, autopct='%1.1f%%', 
                          colors=['skyblue', 'lightcoral'])
            axes[0, 2].set_title('合并数据中的样本来源分布')
        
        # 4. 数据覆盖率分析
        if results['115例'] and results['203例']:
            coverage_data = [
                len(merge_result['sample_cols_115']) if merge_result else 0,
                len(merge_result['sample_cols_203']) if merge_result else 0,
                results['115例']['shape'][0],
                results['203例']['shape'][0]
            ]
            coverage_labels = ['115例\n可用样本', '203例\n可用样本', '115例\n总代谢物', '203例\n总代谢物']
            colors = ['lightblue', 'lightpink', 'darkblue', 'darkred']
            
            bars4 = axes[1, 0].bar(coverage_labels, coverage_data, color=colors)
            axes[1, 0].set_title('数据覆盖率分析')
            axes[1, 0].set_ylabel('数量')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for i, v in enumerate(coverage_data):
                axes[1, 0].text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        # 5. 合并效率分析
        if merge_result and results['115例'] and results['203例']:
            original_total = results['115例']['shape'][0] + results['203例']['shape'][0]
            merged_metabolites = merge_result['common_metabolites_count']
            efficiency = (merged_metabolites / original_total) * 100
            
            efficiency_data = [
                results['115例']['shape'][0],
                results['203例']['shape'][0],
                merged_metabolites,
                original_total - merged_metabolites
            ]
            efficiency_labels = ['115例\n独有', '203例\n独有', '共同\n代谢物', '丢失\n代谢物']
            colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightgray']
            
            axes[1, 1].pie(efficiency_data, labels=efficiency_labels, autopct='%1.1f%%', colors=colors)
            axes[1, 1].set_title(f'代谢物合并效率\n(保留率: {efficiency:.1f}%)')
        
        # 6. 合并前后的数据维度变化
        if merge_result and results['115例'] and results['203例']:
            # 合并前的维度
            dim_before = [
                results['115例']['shape'][0],  # 115例代谢物数
                results['203例']['shape'][0],  # 203例代谢物数
                len(merge_result['sample_cols_115']),  # 115例样本数
                len(merge_result['sample_cols_203'])   # 203例样本数
            ]
            
            # 合并后的维度
            dim_after = [
                merge_result['common_metabolites_count'],  # 合并后代谢物数
                merge_result['common_metabolites_count'],  # 合并后代谢物数（重复显示对比）
                merge_result['total_samples'],  # 合并后总样本数
                merge_result['total_samples']   # 合并后总样本数（重复显示对比）
            ]
            
            x_pos = np.arange(4)
            width = 0.35
            
            bars1 = axes[1, 2].bar(x_pos - width/2, dim_before, width, 
                                  label='合并前', color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
            bars2 = axes[1, 2].bar(x_pos + width/2, dim_after, width, 
                                  label='合并后', color=['darkblue', 'darkred', 'darkgreen', 'orange'])
            
            axes[1, 2].set_title('合并前后的数据维度变化')
            axes[1, 2].set_ylabel('数量')
            axes[1, 2].set_xlabel('数据维度')
            axes[1, 2].set_xticks(x_pos)
            axes[1, 2].set_xticklabels(['115例\n代谢物', '203例\n代谢物', '115例\n样本', '203例\n样本'], rotation=45)
            axes[1, 2].legend()
            
            # 添加数值标签
            for i, (v1, v2) in enumerate(zip(dim_before, dim_after)):
                axes[1, 2].text(i - width/2, v1 + max(dim_before)*0.01, str(v1), ha='center', va='bottom', fontsize=8)
                axes[1, 2].text(i + width/2, v2 + max(dim_after)*0.01, str(v2), ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('metabolite_merge_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n数据合并分析可视化已保存为: metabolite_merge_analysis.png")
        
    except Exception as e:
        print(f"创建可视化时出错: {e}")

def main():
    """主函数"""
    print("开始正确分析和合并代谢物数据...")
    
    # 1. 分析数据结构
    results = analyze_metabolite_data_structure()
    
    if not results or '115例' not in results or '203例' not in results:
        print("数据分析失败")
        return
    
    # 2. 找到共同代谢物
    common_metabolites_info = find_common_metabolites(results)
    
    if not common_metabolites_info:
        print("无法找到共同代谢物")
        return
    
    # 2.5 保存共同代谢物详细信息
    common_metabolites_file = save_common_metabolites_info(results, common_metabolites_info)
    
    # 3. 合并数据集
    merge_result = merge_metabolite_datasets(results, common_metabolites_info)
    
    if not merge_result:
        print("数据合并失败")
        return
    
    # 4. 保存合并后的数据
    save_result = save_merged_data(merge_result, "metabolite_115_203_merged")
    
    # 5. 创建可视化
    create_visualization(results, merge_result)
    
    print("\n=== 合并完成 ===")
    print("输出文件:")
    print(f"1. {common_metabolites_file} - 共同代谢物详细信息")
    if save_result:
        print(f"2. {save_result['full_data_file']} - 完整合并数据")
        print(f"3. {save_result['ml_format_file']} - 机器学习格式数据")
    print("4. metabolite_merge_analysis.png - 数据合并分析可视化")

if __name__ == "__main__":
    main()
