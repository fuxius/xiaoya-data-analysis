#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ2综合分析：详细的性能评估报告
包含AI模型基准性能和医生在不同分组下的详细准确率分析

作者: AICare研究团队
日期: 2025年9月
"""

import pandas as pd
import numpy as np
import json
import re
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, 
                           precision_recall_curve, auc)
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载所有必要数据"""
    print("正在加载数据文件...")
    
    # 加载合并数据集
    try:
        df = pd.read_excel('results/datasets/merged_dataset_simple.xlsx')
        print(f"✓ 成功加载合并数据集: {df.shape}")
    except:
        df = pd.read_excel('results/merged_dataset_simple.xlsx')
        print(f"✓ 成功加载合并数据集: {df.shape}")
    
    # 加载年资分类
    with open('results/participants/seniority_classification.json', 'r', encoding='utf-8') as f:
        seniority_data = json.load(f)
    print("✓ 成功加载年资分类数据")
    
    # 加载患者结果数据
    patient_data = pd.read_csv('data/patient_last_risk_summary_ordered.csv')
    print(f"✓ 成功加载患者结果数据: {patient_data.shape}")
    
    return df, seniority_data, patient_data

def add_grouping_variables(df, seniority_data):
    """添加分组变量"""
    print("\n正在添加分组变量...")
    
    # 添加年资分类
    df['年资分类'] = df['ID'].map(lambda x: seniority_data['participant_classification'].get(x, {}).get('seniority_level', '未知'))
    
    # 添加AI使用顺序
    df['ai_first'] = df['是否先使用AI分析系统'] == '是'
    
    # 添加医院分组
    df['hospital'] = df['ID'].str[:2]
    
    print(f"年资分类分布: {df['年资分类'].value_counts().to_dict()}")
    print(f"AI使用顺序分布: {df['ai_first'].value_counts().to_dict()}")
    print(f"医院分布: {df['hospital'].value_counts().to_dict()}")
    print(f"科室分布: {df['科室'].value_counts().to_dict()}")
    
    return df

def find_optimal_threshold(y_true, y_scores):
    """为给定数据集找到最优阈值"""
    # 使用所有预测值作为候选阈值
    thresholds = sorted(set(y_scores))
    
    best_acc = 0
    best_threshold = 50.0  # 默认50%
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = threshold
    
    return best_threshold, best_acc

def calculate_ai_model_performance(patient_data):
    """计算AI模型的详细基准性能（使用最优阈值）"""
    print("\n=== 1. AI模型基准性能评估（使用最优阈值）===")
    
    results = []
    
    # 按数据集分组计算
    for dataset in patient_data['dataset'].unique():
        dataset_data = patient_data[patient_data['dataset'] == dataset]
        
        if len(dataset_data) > 0:
            y_true = dataset_data['outcome'].values
            y_scores = dataset_data['last_risk_value'].values
            
            # 找到最优阈值
            optimal_threshold, optimal_accuracy = find_optimal_threshold(y_true, y_scores)
            
            # 使用最优阈值进行预测
            y_pred = (y_scores >= optimal_threshold).astype(int)
            
            # 计算基本指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0) if np.sum(y_true) > 0 else np.nan
            recall = recall_score(y_true, y_pred, zero_division=0) if np.sum(y_true) > 0 else np.nan
            f1 = f1_score(y_true, y_pred, zero_division=0) if np.sum(y_true) > 0 else np.nan
            
            # 计算特异性
            specificity = np.nan
            if np.sum(1 - y_true) > 0:
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # 计算AUROC和AUPRC
            auroc = np.nan
            auprc = np.nan
            if len(np.unique(y_true)) > 1:
                try:
                    auroc = roc_auc_score(y_true, y_scores)
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
                    auprc = auc(recall_curve, precision_curve)
                except:
                    pass
            
            results.append({
                'dataset': dataset,
                'performance_type': 'AI模型基准',
                'group_type': '数据集',
                'group_name': dataset,
                'n_samples': len(dataset_data),
                'n_positive': sum(y_true),
                'n_negative': len(y_true) - sum(y_true),
                'positive_rate': sum(y_true) / len(y_true),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'auroc': auroc,
                'auprc': auprc
            })
            
            print(f"{dataset}: 最优阈值={optimal_threshold:.2f}, 准确率={accuracy:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}")
    
    # 总体性能（加权平均最优性能）
    y_true_all = patient_data['outcome'].values
    y_scores_all = patient_data['last_risk_value'].values
    
    # 计算加权平均的最优准确率
    total_samples = len(patient_data)
    weighted_optimal_acc = sum(r['accuracy'] * r['n_samples'] for r in results) / total_samples
    
    # 为了保持一致性，这里仍然计算其他指标（使用50%阈值作为参考）
    y_pred_all = (y_scores_all >= 50).astype(int)
    precision_all = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall_all = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1_all = f1_score(y_true_all, y_pred_all, zero_division=0)
    
    tn_all = np.sum((y_true_all == 0) & (y_pred_all == 0))
    fp_all = np.sum((y_true_all == 0) & (y_pred_all == 1))
    specificity_all = tn_all / (tn_all + fp_all) if (tn_all + fp_all) > 0 else 0
    
    auroc_all = roc_auc_score(y_true_all, y_scores_all)
    precision_curve_all, recall_curve_all, _ = precision_recall_curve(y_true_all, y_scores_all)
    auprc_all = auc(recall_curve_all, precision_curve_all)
    
    results.append({
        'dataset': '总体',
        'performance_type': 'AI模型基准',
        'group_type': '总体',
        'group_name': '所有数据集',
        'n_samples': len(patient_data),
        'n_positive': sum(y_true_all),
        'n_negative': len(y_true_all) - sum(y_true_all),
        'positive_rate': sum(y_true_all) / len(y_true_all),
        'accuracy': weighted_optimal_acc,
        'precision': precision_all,
        'recall': recall_all,
        'f1_score': f1_all,
        'specificity': specificity_all,
        'auroc': auroc_all,
        'auprc': auprc_all
    })
    
    print(f"总体: 加权最优准确率={weighted_optimal_acc:.4f}, 精确率={precision_all:.4f}, 召回率={recall_all:.4f}")
    
    return pd.DataFrame(results)

def convert_risk_assessment_to_score(risk_text):
    """将风险评估文本转换为数值分数"""
    if pd.isna(risk_text) or risk_text == '(跳过)':
        return np.nan
    
    risk_mapping = {
        '0-25%': 12.5,
        '25-50%': 37.5,
        '50-75%': 62.5,
        '75-100%': 87.5
    }
    
    for key, value in risk_mapping.items():
        if key in str(risk_text):
            return value
    
    return np.nan

def extract_clinician_assessments(df, patient_data):
    """提取医生的风险评估数据"""
    print("\n正在提取医生风险评估数据...")
    
    # 找到风险评估相关的列
    risk_assessment_cols = []
    for col in df.columns:
        if '您如何评估患者' in col and '死亡风险' in col:
            risk_assessment_cols.append(col)
        elif '您如何评估孕妇' in col and '早产' in col:
            risk_assessment_cols.append(col)
    
    print(f"找到风险评估列数: {len(risk_assessment_cols)}")
    
    # 创建患者编号到数据集的映射
    patient_mapping = {}
    for idx, row in patient_data.iterrows():
        key = f"{row['dataset']}_{row['patient_number']}"
        patient_mapping[key] = {
            'dataset': row['dataset'],
            'patient_number': row['patient_number'],
            'outcome': row['outcome']
        }
    
    # 提取每个医生对每个患者的评估
    clinician_assessments = []
    
    for idx, row in df.iterrows():
        participant_id = row['ID']
        department = row['科室']
        
        # 根据科室确定患者映射
        if 'Nephrology' in department:
            if 'XY' in participant_id:
                dataset_name = '徐医肾内科'
            elif 'BC' in participant_id or 'BS' in participant_id:
                dataset_name = '北医肾内科'
            else:
                continue
        elif 'Obstetrics' in department:
            dataset_name = '北医多胎'
        else:
            continue
        
        # 提取该参与者的所有风险评估
        for i, col in enumerate(risk_assessment_cols):
            if pd.notna(row[col]) and row[col] != '(跳过)':
                # 从列名中提取患者编号
                if '患者' in col:
                    patient_match = re.search(r'患者(\d+)', col)
                    if patient_match:
                        patient_num = int(patient_match.group(1))
                    else:
                        continue
                elif '孕妇' in col:
                    patient_match = re.search(r'孕妇(\d+)', col)
                    if patient_match:
                        patient_num = int(patient_match.group(1))
                    else:
                        continue
                else:
                    continue
                
                # 查找对应的真实结果
                patient_key = f"{dataset_name}_{patient_num}"
                if patient_key in patient_mapping:
                    risk_score = convert_risk_assessment_to_score(row[col])
                    
                    if not pd.isna(risk_score):
                        clinician_assessments.append({
                            'participant_id': participant_id,
                            'department': department,
                            'dataset': dataset_name,
                            'patient_number': patient_num,
                            'assessment_column': col,
                            'risk_text': row[col],
                            'risk_score': risk_score,
                            'true_outcome': patient_mapping[patient_key]['outcome'],
                            'case_order': patient_num
                        })
    
    clinician_df = pd.DataFrame(clinician_assessments)
    print(f"提取到医生评估记录: {len(clinician_df)}条")
    
    return clinician_df

def assign_ai_conditions(df, clinician_df):
    """分配AI辅助条件"""
    print("\n正在分配AI辅助条件...")
    
    condition_data = []
    
    for idx, row in clinician_df.iterrows():
        participant_id = row['participant_id']
        case_order = row['case_order']
        
        # 获取该参与者的AI使用顺序
        participant_info = df[df['ID'] == participant_id].iloc[0]
        ai_first = participant_info['ai_first']
        
        # 确定条件
        if ai_first:
            condition = 'AI辅助' if case_order <= 5 else '无辅助'
        else:
            condition = '无辅助' if case_order <= 5 else 'AI辅助'
        
        condition_data.append({
            **row.to_dict(),
            'ai_first': ai_first,
            'condition': condition
        })
    
    condition_df = pd.DataFrame(condition_data)
    
    print("条件分配统计:")
    print(condition_df['condition'].value_counts())
    
    return condition_df

def calculate_individual_performance(condition_df, df):
    """计算每个医生的个体性能（包含完整指标）"""
    print("\n正在计算医生个体性能...")
    
    # 添加分组信息
    condition_df = condition_df.merge(
        df[['ID', '年资分类', 'hospital']].rename(columns={'ID': 'participant_id'}),
        on='participant_id',
        how='left'
    )
    
    individual_performance = []
    
    for participant_id in condition_df['participant_id'].unique():
        participant_data = condition_df[condition_df['participant_id'] == participant_id]
        participant_info = df[df['ID'] == participant_id].iloc[0]
        
        # 为每种条件计算性能
        for condition in ['AI辅助', '无辅助']:
            condition_assessments = participant_data[participant_data['condition'] == condition]
            
            if len(condition_assessments) >= 3:
                y_true = condition_assessments['true_outcome'].values
                y_scores = condition_assessments['risk_score'].values
                y_pred = (y_scores >= 50).astype(int)
                
                # 计算基本指标
                accuracy = accuracy_score(y_true, y_pred)
                
                # 计算其他指标
                precision_val = np.nan
                recall_val = np.nan
                f1_score_val = np.nan
                specificity_val = np.nan
                auroc_val = np.nan
                auprc_val = np.nan
                
                # 精确率、召回率、F1 (需要正样本)
                if np.sum(y_true) > 0:
                    precision_val = precision_score(y_true, y_pred, zero_division=0)
                    recall_val = recall_score(y_true, y_pred, zero_division=0)
                    f1_score_val = f1_score(y_true, y_pred, zero_division=0)
                
                # 特异性 (需要负样本)
                if np.sum(1 - y_true) > 0:
                    tn = np.sum((y_true == 0) & (y_pred == 0))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # AUROC, AUPRC (需要正负样本混合)
                if len(np.unique(y_true)) > 1:
                    try:
                        auroc_val = roc_auc_score(y_true, y_scores)
                        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
                        auprc_val = auc(recall_curve, precision_curve)
                    except Exception:
                        pass
                
                individual_performance.append({
                    'participant_id': participant_id,
                    'department': participant_info['科室'],
                    'seniority': participant_info['年资分类'],
                    'hospital': participant_info['hospital'],
                    'condition': condition,
                    'n_cases': len(condition_assessments),
                    'n_positive': np.sum(y_true),
                    'n_negative': len(y_true) - np.sum(y_true),
                    'accuracy': accuracy,
                    'precision': precision_val,
                    'recall': recall_val,
                    'f1_score': f1_score_val,
                    'specificity': specificity_val,
                    'auroc': auroc_val,
                    'auprc': auprc_val
                })
    
    return pd.DataFrame(individual_performance)

def calculate_group_statistics(individual_performance_df):
    """计算分组统计（包含所有指标）"""
    print("\n正在计算分组统计...")
    
    group_stats = []
    
    # 定义分组
    groupings = [
        ('总体', '所有医生', lambda df: df),
        ('年资', '高年资', lambda df: df[df['seniority'] == '高年资']),
        ('年资', '低年资', lambda df: df[df['seniority'] == '低年资']),
        ('科室', 'XY-Nephrology', lambda df: df[df['department'] == 'XY-Nephrology']),
        ('科室', 'BC-Obstetrics', lambda df: df[df['department'] == 'BC-Obstetrics']),
        ('科室', 'BS-Nephrology', lambda df: df[df['department'] == 'BS-Nephrology']),
        ('医院', 'XY', lambda df: df[df['hospital'] == 'XY']),
        ('医院', 'BC', lambda df: df[df['hospital'] == 'BC']),
        ('医院', 'BS', lambda df: df[df['hospital'] == 'BS'])
    ]
    
    # 要统计的指标
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auroc', 'auprc']
    
    for group_type, group_name, filter_func in groupings:
        group_data = filter_func(individual_performance_df)
        
        if len(group_data) > 0:
            # 为每种条件计算统计
            for condition in ['AI辅助', '无辅助']:
                condition_data = group_data[group_data['condition'] == condition]
                
                if len(condition_data) > 0:
                    stats_dict = {
                        'group_type': group_type,
                        'group_name': group_name,
                        'condition': condition,
                        'n_participants': len(condition_data)
                    }
                    
                    # 为每个指标计算统计
                    for metric in metrics:
                        if metric in condition_data.columns:
                            values = condition_data[metric].dropna()  # 移除NaN值
                            if len(values) > 0:
                                stats_dict.update({
                                    f'{metric}_mean': values.mean(),
                                    f'{metric}_std': values.std(),
                                    f'{metric}_min': values.min(),
                                    f'{metric}_max': values.max(),
                                    f'{metric}_median': values.median(),
                                    f'{metric}_count': len(values)  # 有效值数量
                                })
                            else:
                                stats_dict.update({
                                    f'{metric}_mean': np.nan,
                                    f'{metric}_std': np.nan,
                                    f'{metric}_min': np.nan,
                                    f'{metric}_max': np.nan,
                                    f'{metric}_median': np.nan,
                                    f'{metric}_count': 0
                                })
                    
                    group_stats.append(stats_dict)
    
    return pd.DataFrame(group_stats)

def perform_statistical_tests(individual_performance_df):
    """执行统计检验"""
    print("\n正在执行统计检验...")
    
    test_results = []
    
    # 定义分组
    groupings = [
        ('总体', '所有医生', lambda df: df),
        ('年资', '高年资', lambda df: df[df['seniority'] == '高年资']),
        ('年资', '低年资', lambda df: df[df['seniority'] == '低年资']),
        ('科室', 'XY-Nephrology', lambda df: df[df['department'] == 'XY-Nephrology']),
        ('科室', 'BC-Obstetrics', lambda df: df[df['department'] == 'BC-Obstetrics']),
        ('科室', 'BS-Nephrology', lambda df: df[df['department'] == 'BS-Nephrology']),
        ('医院', 'XY', lambda df: df[df['hospital'] == 'XY']),
        ('医院', 'BC', lambda df: df[df['hospital'] == 'BC']),
        ('医院', 'BS', lambda df: df[df['hospital'] == 'BS'])
    ]
    
    for group_type, group_name, filter_func in groupings:
        group_data = filter_func(individual_performance_df)
        
        # 获取配对数据
        paired_data = []
        participant_ids = group_data['participant_id'].unique()
        
        for pid in participant_ids:
            pid_data = group_data[group_data['participant_id'] == pid]
            ai_data = pid_data[pid_data['condition'] == 'AI辅助']
            no_ai_data = pid_data[pid_data['condition'] == '无辅助']
            
            if len(ai_data) > 0 and len(no_ai_data) > 0:
                paired_data.append({
                    'participant_id': pid,
                    'ai_accuracy': ai_data.iloc[0]['accuracy'],
                    'no_ai_accuracy': no_ai_data.iloc[0]['accuracy']
                })
        
        if len(paired_data) > 1:
            ai_accuracies = [p['ai_accuracy'] for p in paired_data]
            no_ai_accuracies = [p['no_ai_accuracy'] for p in paired_data]
            
            # 执行配对t检验
            t_stat, p_value = stats.ttest_rel(ai_accuracies, no_ai_accuracies)
            
            # 计算效应量
            differences = np.array(ai_accuracies) - np.array(no_ai_accuracies)
            cohens_d = differences.mean() / differences.std(ddof=1) if differences.std(ddof=1) != 0 else 0
            
            test_results.append({
                'group_type': group_type,
                'group_name': group_name,
                'n_pairs': len(paired_data),
                'ai_mean': np.mean(ai_accuracies),
                'ai_std': np.std(ai_accuracies, ddof=1),
                'no_ai_mean': np.mean(no_ai_accuracies),
                'no_ai_std': np.std(no_ai_accuracies, ddof=1),
                'mean_difference': np.mean(differences),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significance': 'Yes' if p_value < 0.05 else 'No'
            })
            
            print(f"{group_type}-{group_name}: n={len(paired_data)}, 差异={np.mean(differences):+.4f}, p={p_value:.4f}")
    
    return pd.DataFrame(test_results)

def create_comprehensive_comparison(ai_model_results, group_stats, test_results):
    """创建综合对比表"""
    print("\n正在创建综合对比表...")
    
    comparison_data = []
    
    # AI模型基准
    ai_overall = ai_model_results[ai_model_results['group_name'] == '所有数据集'].iloc[0]
    comparison_data.append({
        'performance_type': 'AI模型基准',
        'group_type': '总体',
        'group_name': '所有数据集',
        'accuracy': ai_overall['accuracy'],
        'std': np.nan,
        'n_samples': ai_overall['n_samples'],
        'description': 'AI模型在30个案例上的基准准确率'
    })
    
    # 医生表现
    for condition in ['无辅助', 'AI辅助']:
        condition_stats = group_stats[group_stats['condition'] == condition]
        
        for idx, row in condition_stats.iterrows():
            comparison_data.append({
                'performance_type': f'医生{condition}',
                'group_type': row['group_type'],
                'group_name': row['group_name'],
                'accuracy': row['accuracy_mean'],
                'std': row['accuracy_std'],
                'n_samples': row['n_participants'],
                'description': f"{row['group_name']}医生在{condition}条件下的准确率"
            })
    
    # 添加统计检验结果
    comparison_df = pd.DataFrame(comparison_data)
    
    # 合并检验结果
    for idx, test_row in test_results.iterrows():
        # 找到对应的AI辅助和无辅助行
        mask_ai = ((comparison_df['performance_type'] == '医生AI辅助') & 
                  (comparison_df['group_type'] == test_row['group_type']) & 
                  (comparison_df['group_name'] == test_row['group_name']))
        
        mask_no_ai = ((comparison_df['performance_type'] == '医生无辅助') & 
                     (comparison_df['group_type'] == test_row['group_type']) & 
                     (comparison_df['group_name'] == test_row['group_name']))
        
        # 添加统计信息
        if mask_ai.any():
            comparison_df.loc[mask_ai, 't_statistic'] = test_row['t_statistic']
            comparison_df.loc[mask_ai, 'p_value'] = test_row['p_value']
            comparison_df.loc[mask_ai, 'cohens_d'] = test_row['cohens_d']
            comparison_df.loc[mask_ai, 'significance'] = test_row['significance']
        
        if mask_no_ai.any():
            comparison_df.loc[mask_no_ai, 't_statistic'] = test_row['t_statistic']
            comparison_df.loc[mask_no_ai, 'p_value'] = test_row['p_value']
            comparison_df.loc[mask_no_ai, 'cohens_d'] = test_row['cohens_d']
            comparison_df.loc[mask_no_ai, 'significance'] = test_row['significance']
    
    return comparison_df

def generate_summary_report(ai_model_results, test_results):
    """生成总结报告"""
    print("\n正在生成总结报告...")
    
    report = []
    
    # AI模型基准性能
    ai_overall = ai_model_results[ai_model_results['group_name'] == '所有数据集'].iloc[0]
    report.append(f"AI模型基准准确率: {ai_overall['accuracy']:.4f}")
    
    # 总体效果
    overall_test = test_results[test_results['group_name'] == '所有医生'].iloc[0]
    report.append(f"总体效果: AI辅助使医生准确率{overall_test['mean_difference']:+.4f}")
    report.append(f"统计显著性: {'显著' if overall_test['significance'] == 'Yes' else '不显著'} (p={overall_test['p_value']:.4f})")
    report.append(f"效应量: Cohen's d = {overall_test['cohens_d']:.4f}")
    
    # 年资差异
    seniority_tests = test_results[test_results['group_type'] == '年资']
    if len(seniority_tests) > 0:
        report.append("\n年资差异:")
        for idx, row in seniority_tests.iterrows():
            report.append(f"  {row['group_name']}: 准确率提升{row['mean_difference']:+.4f} (p={row['p_value']:.4f})")
    
    # 科室差异
    dept_tests = test_results[test_results['group_type'] == '科室']
    if len(dept_tests) > 0:
        report.append("\n科室差异:")
        for idx, row in dept_tests.iterrows():
            report.append(f"  {row['group_name']}: 准确率提升{row['mean_difference']:+.4f} (p={row['p_value']:.4f})")
    
    # 医院差异
    hospital_tests = test_results[test_results['group_type'] == '医院']
    if len(hospital_tests) > 0:
        report.append("\n医院差异:")
        for idx, row in hospital_tests.iterrows():
            report.append(f"  医院{row['group_name']}: 准确率提升{row['mean_difference']:+.4f} (p={row['p_value']:.4f})")
    
    return report

def main():
    """主函数"""
    print("=== RQ2: 综合性能分析 ===")
    
    # 加载数据
    df, seniority_data, patient_data = load_data()
    df = add_grouping_variables(df, seniority_data)
    
    # 1. 计算AI模型基准性能
    ai_model_results = calculate_ai_model_performance(patient_data)
    
    # 2. 提取医生评估数据
    clinician_assessments = extract_clinician_assessments(df, patient_data)
    condition_df = assign_ai_conditions(df, clinician_assessments)
    
    # 3. 计算个体性能
    individual_performance_df = calculate_individual_performance(condition_df, df)
    
    # 4. 计算分组统计
    group_stats = calculate_group_statistics(individual_performance_df)
    
    # 5. 执行统计检验
    test_results = perform_statistical_tests(individual_performance_df)
    
    # 6. 创建综合对比
    comprehensive_comparison = create_comprehensive_comparison(ai_model_results, group_stats, test_results)
    
    # 7. 生成报告
    summary_report = generate_summary_report(ai_model_results, test_results)
    
    # 8. 保存所有结果
    print("\n正在保存结果文件...")
    
    output_dir = 'results/rq2_performance_analysis'
    
    # AI模型详细性能
    ai_model_results.to_csv(f'{output_dir}/ai_model_detailed_performance.csv', index=False, encoding='utf-8-sig')
    print("✓ AI模型详细性能已保存")
    
    # 医生个体性能
    individual_performance_df.to_csv(f'{output_dir}/clinician_individual_performance.csv', index=False, encoding='utf-8-sig')
    print("✓ 医生个体性能已保存")
    
    # 分组统计
    group_stats.to_csv(f'{output_dir}/clinician_group_statistics.csv', index=False, encoding='utf-8-sig')
    print("✓ 分组统计已保存")
    
    # 统计检验结果
    test_results.to_csv(f'{output_dir}/statistical_test_results.csv', index=False, encoding='utf-8-sig')
    print("✓ 统计检验结果已保存")
    
    # 综合对比表
    comprehensive_comparison.to_csv(f'{output_dir}/comprehensive_performance_comparison.csv', index=False, encoding='utf-8-sig')
    print("✓ 综合对比表已保存")
    
    # 创建配对数据表（便于进一步分析）
    paired_data = []
    for pid in individual_performance_df['participant_id'].unique():
        pid_data = individual_performance_df[individual_performance_df['participant_id'] == pid]
        ai_data = pid_data[pid_data['condition'] == 'AI辅助']
        no_ai_data = pid_data[pid_data['condition'] == '无辅助']
        
        if len(ai_data) > 0 and len(no_ai_data) > 0:
            paired_data.append({
                'participant_id': pid,
                'department': ai_data.iloc[0]['department'],
                'seniority': ai_data.iloc[0]['seniority'],
                'hospital': ai_data.iloc[0]['hospital'],
                'accuracy_ai': ai_data.iloc[0]['accuracy'],
                'accuracy_no_ai': no_ai_data.iloc[0]['accuracy'],
                'accuracy_difference': ai_data.iloc[0]['accuracy'] - no_ai_data.iloc[0]['accuracy']
            })
    
    paired_df = pd.DataFrame(paired_data)
    paired_df.to_csv(f'{output_dir}/paired_performance_data.csv', index=False, encoding='utf-8-sig')
    print("✓ 配对性能数据已保存")
    
    # 保存总结报告
    with open(f'{output_dir}/summary_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== RQ2: 综合性能分析报告 ===\n\n")
        for line in summary_report:
            f.write(f"{line}\n")
    print("✓ 总结报告已保存")
    
    # 显示关键结果
    print(f"\n=== 关键结果概览 ===")
    print(f"有效参与者数量: {len(individual_performance_df['participant_id'].unique())}")
    
    # 显示AI模型性能
    ai_overall = ai_model_results[ai_model_results['group_name'] == '所有数据集'].iloc[0]
    print(f"AI模型基准准确率: {ai_overall['accuracy']:.4f}")
    
    # 显示总体效果
    overall_test = test_results[test_results['group_name'] == '所有医生'].iloc[0]
    print(f"医生无AI辅助准确率: {overall_test['no_ai_mean']:.4f} ± {overall_test['no_ai_std']:.4f}")
    print(f"医生有AI辅助准确率: {overall_test['ai_mean']:.4f} ± {overall_test['ai_std']:.4f}")
    print(f"平均准确率提升: {overall_test['mean_difference']:+.4f}")
    print(f"统计显著性: {'是' if overall_test['significance'] == 'Yes' else '否'} (p={overall_test['p_value']:.4f})")
    
    # 显示分组结果
    print(f"\n=== 分组结果 ===")
    for idx, row in test_results.iterrows():
        if row['group_type'] != '总体':
            print(f"{row['group_type']}-{row['group_name']}: 提升{row['mean_difference']:+.4f} (p={row['p_value']:.4f})")
    
    print(f"\n所有结果文件已保存到: {output_dir}/")

if __name__ == "__main__":
    main()