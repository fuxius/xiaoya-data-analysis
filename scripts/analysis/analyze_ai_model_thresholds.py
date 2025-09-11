#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI模型基准性能详细分析：阈值优化研究

详细分析AI模型在不同风险阈值下的性能表现，
包括当前使用的50%阈值和各数据集的最优阈值。

作者: AICare研究团队
日期: 2025年9月
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, precision_recall_curve, 
                           roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

def load_patient_data():
    """加载患者数据"""
    print("=== AI模型基准性能详细分析 ===\n")
    
    patient_data = pd.read_csv('data/patient_last_risk_summary_ordered.csv')
    print(f"✓ 加载患者数据: {len(patient_data)}个案例")
    
    # 显示数据概览
    print("\n📊 数据集分布:")
    for dataset in patient_data['dataset'].unique():
        subset = patient_data[patient_data['dataset'] == dataset]
        n_positive = sum(subset['outcome'])
        n_negative = len(subset) - n_positive
        print(f"  {dataset}: {len(subset)}例 (正样本:{n_positive}, 负样本:{n_negative})")
    
    return patient_data

def explain_current_evaluation_method():
    """解释当前的评估方法"""
    print("\n🔍 当前AI模型评估方法详解:")
    print("=" * 50)
    
    print("1. **数据来源**: data/patient_last_risk_summary_ordered.csv")
    print("   - 包含30个患者案例的AI风险预测值和真实结局")
    print("   - 分为3个数据集：北医多胎(10例)、北医肾内科(10例)、徐医肾内科(10例)")
    
    print("\n2. **当前使用的分类阈值**: 50%")
    print("   - 如果 AI预测风险 >= 50%，则预测为正类(高风险)")
    print("   - 如果 AI预测风险 < 50%，则预测为负类(低风险)")
    
    print("\n3. **评估指标计算**:")
    print("   - 准确率 = (正确预测数) / (总预测数)")
    print("   - 精确率 = (真正例) / (真正例 + 假正例)")
    print("   - 召回率 = (真正例) / (真正例 + 假负例)")
    print("   - F1分数 = 2 * (精确率 * 召回率) / (精确率 + 召回率)")
    
    print("\n4. **问题分析**:")
    print("   - 50%阈值是一个通用阈值，可能不是每个数据集的最优阈值")
    print("   - 不同数据集的风险分布不同，应该使用不同的最优阈值")
    print("   - 需要通过ROC曲线和PR曲线找到最优阈值")

def calculate_metrics_for_threshold(y_true, y_scores, threshold):
    """计算指定阈值下的所有指标"""
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'n_samples': len(y_true),
        'n_positive': np.sum(y_true),
        'n_negative': len(y_true) - np.sum(y_true)
    }
    
    # 计算混淆矩阵元素
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    metrics.update({
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    })
    
    return metrics

def find_optimal_thresholds(patient_data):
    """找到每个数据集的最优阈值"""
    print("\n🎯 寻找最优阈值:")
    print("=" * 50)
    
    optimal_results = []
    
    # 测试的阈值范围
    thresholds_to_test = np.arange(0.05, 0.95, 0.05)
    
    for dataset in patient_data['dataset'].unique():
        dataset_data = patient_data[patient_data['dataset'] == dataset]
        y_true = dataset_data['outcome'].values
        y_scores = dataset_data['last_risk_value'].values
        
        print(f"\n📈 分析数据集: {dataset}")
        print(f"   样本数: {len(dataset_data)}, 正样本: {sum(y_true)}, 负样本: {len(y_true) - sum(y_true)}")
        
        # 如果只有一类样本，跳过
        if len(np.unique(y_true)) == 1:
            print(f"   ⚠️  只有一类样本，无法计算最优阈值")
            continue
        
        # 测试所有阈值
        results = []
        for threshold in thresholds_to_test:
            metrics = calculate_metrics_for_threshold(y_true, y_scores, threshold)
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # 找到各种指标的最优阈值
        best_accuracy_idx = results_df['accuracy'].idxmax()
        best_f1_idx = results_df['f1_score'].idxmax()
        
        best_accuracy_threshold = results_df.loc[best_accuracy_idx, 'threshold']
        best_f1_threshold = results_df.loc[best_f1_idx, 'threshold']
        
        # 计算当前50%阈值的性能
        current_metrics = calculate_metrics_for_threshold(y_true, y_scores, 0.5)
        best_accuracy_metrics = results_df.loc[best_accuracy_idx]
        best_f1_metrics = results_df.loc[best_f1_idx]
        
        print(f"   当前阈值(50%): 准确率={current_metrics['accuracy']:.4f}, F1={current_metrics['f1_score']:.4f}")
        print(f"   最优准确率阈值({best_accuracy_threshold:.2f}): 准确率={best_accuracy_metrics['accuracy']:.4f}")
        print(f"   最优F1阈值({best_f1_threshold:.2f}): F1={best_f1_metrics['f1_score']:.4f}")
        
        # 保存结果
        optimal_results.append({
            'dataset': dataset,
            'current_threshold': 0.5,
            'current_accuracy': current_metrics['accuracy'],
            'current_f1': current_metrics['f1_score'],
            'optimal_accuracy_threshold': best_accuracy_threshold,
            'optimal_accuracy': best_accuracy_metrics['accuracy'],
            'optimal_f1_threshold': best_f1_threshold,
            'optimal_f1': best_f1_metrics['f1_score'],
            'accuracy_improvement': best_accuracy_metrics['accuracy'] - current_metrics['accuracy'],
            'f1_improvement': best_f1_metrics['f1_score'] - current_metrics['f1_score'],
            'all_results': results_df
        })
    
    return optimal_results

def analyze_dataset_specific_thresholds(patient_data):
    """分析各数据集推荐的特定阈值"""
    print("\n📋 各数据集推荐阈值分析:")
    print("=" * 50)
    
    # 从风险阈值文件读取推荐阈值
    recommended_thresholds = {
        '北医多胎': 0.34,
        '北医肾内科': 0.522,
        '徐医肾内科': 0.503
    }
    
    recommended_results = []
    
    for dataset in patient_data['dataset'].unique():
        dataset_data = patient_data[patient_data['dataset'] == dataset]
        y_true = dataset_data['outcome'].values
        y_scores = dataset_data['last_risk_value'].values
        
        print(f"\n🎯 数据集: {dataset}")
        
        # 当前50%阈值性能
        current_metrics = calculate_metrics_for_threshold(y_true, y_scores, 0.5)
        
        # 推荐阈值性能
        recommended_threshold = recommended_thresholds.get(dataset, 0.5)
        recommended_metrics = calculate_metrics_for_threshold(y_true, y_scores, recommended_threshold)
        
        print(f"   推荐阈值: {recommended_threshold:.3f}")
        print(f"   当前阈值(50%): 准确率={current_metrics['accuracy']:.4f}, 精确率={current_metrics['precision']:.4f}, 召回率={current_metrics['recall']:.4f}")
        print(f"   推荐阈值({recommended_threshold:.1%}): 准确率={recommended_metrics['accuracy']:.4f}, 精确率={recommended_metrics['precision']:.4f}, 召回率={recommended_metrics['recall']:.4f}")
        print(f"   准确率提升: {recommended_metrics['accuracy'] - current_metrics['accuracy']:+.4f}")
        
        recommended_results.append({
            'dataset': dataset,
            'recommended_threshold': recommended_threshold,
            'current_accuracy': current_metrics['accuracy'],
            'recommended_accuracy': recommended_metrics['accuracy'],
            'accuracy_improvement': recommended_metrics['accuracy'] - current_metrics['accuracy'],
            'current_metrics': current_metrics,
            'recommended_metrics': recommended_metrics
        })
    
    return recommended_results

def create_threshold_analysis_tables(optimal_results, output_dir):
    """创建阈值分析表格"""
    print("\n📊 生成阈值分析表格...")
    
    # 创建详细的阈值性能表
    all_threshold_results = []
    
    for result in optimal_results:
        dataset = result['dataset']
        results_df = result['all_results']
        
        # 为每个阈值添加数据集标识
        for _, row in results_df.iterrows():
            all_threshold_results.append({
                'dataset': dataset,
                **row.to_dict()
            })
    
    # 保存所有阈值的详细结果
    threshold_details_df = pd.DataFrame(all_threshold_results)
    threshold_details_df.to_csv(f'{output_dir}/threshold_details_all.csv', index=False, encoding='utf-8-sig')
    
    print("✓ 阈值分析表格已保存")

def generate_comprehensive_report(optimal_results, recommended_results, patient_data, output_dir):
    """生成综合分析报告"""
    print("\n📝 生成综合分析报告...")
    
    # 创建详细的结果表格
    detailed_results = []
    
    # 总体性能对比
    for dataset in patient_data['dataset'].unique():
        dataset_data = patient_data[patient_data['dataset'] == dataset]
        y_true = dataset_data['outcome'].values
        y_scores = dataset_data['last_risk_value'].values
        
        # 当前50%阈值
        current_metrics = calculate_metrics_for_threshold(y_true, y_scores, 0.5)
        
        # 找到对应的最优结果
        optimal_result = next((r for r in optimal_results if r['dataset'] == dataset), None)
        recommended_result = next((r for r in recommended_results if r['dataset'] == dataset), None)
        
        # 添加当前阈值结果
        detailed_results.append({
            'dataset': dataset,
            'threshold_type': '当前使用(50%)',
            'threshold': 0.5,
            'accuracy': current_metrics['accuracy'],
            'precision': current_metrics['precision'],
            'recall': current_metrics['recall'],
            'f1_score': current_metrics['f1_score'],
            'specificity': current_metrics['specificity'],
            'tp': current_metrics['tp'],
            'tn': current_metrics['tn'],
            'fp': current_metrics['fp'],
            'fn': current_metrics['fn']
        })
        
        # 添加最优准确率阈值结果
        if optimal_result:
            best_acc_metrics = calculate_metrics_for_threshold(
                y_true, y_scores, optimal_result['optimal_accuracy_threshold'])
            detailed_results.append({
                'dataset': dataset,
                'threshold_type': '最优准确率',
                'threshold': optimal_result['optimal_accuracy_threshold'],
                'accuracy': best_acc_metrics['accuracy'],
                'precision': best_acc_metrics['precision'],
                'recall': best_acc_metrics['recall'],
                'f1_score': best_acc_metrics['f1_score'],
                'specificity': best_acc_metrics['specificity'],
                'tp': best_acc_metrics['tp'],
                'tn': best_acc_metrics['tn'],
                'fp': best_acc_metrics['fp'],
                'fn': best_acc_metrics['fn']
            })
        
        # 添加推荐阈值结果
        if recommended_result:
            rec_metrics = recommended_result['recommended_metrics']
            detailed_results.append({
                'dataset': dataset,
                'threshold_type': '数据集推荐',
                'threshold': recommended_result['recommended_threshold'],
                'accuracy': rec_metrics['accuracy'],
                'precision': rec_metrics['precision'],
                'recall': rec_metrics['recall'],
                'f1_score': rec_metrics['f1_score'],
                'specificity': rec_metrics['specificity'],
                'tp': rec_metrics['tp'],
                'tn': rec_metrics['tn'],
                'fp': rec_metrics['fp'],
                'fn': rec_metrics['fn']
            })
    
    # 保存详细结果
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f'{output_dir}/ai_model_threshold_analysis.csv', index=False, encoding='utf-8-sig')
    
    # 创建总结表
    summary_results = []
    for dataset in patient_data['dataset'].unique():
        dataset_results = detailed_df[detailed_df['dataset'] == dataset]
        
        current_acc = dataset_results[dataset_results['threshold_type'] == '当前使用(50%)']['accuracy'].iloc[0]
        optimal_results_subset = dataset_results[dataset_results['threshold_type'] == '最优准确率']
        recommended_results_subset = dataset_results[dataset_results['threshold_type'] == '数据集推荐']
        
        optimal_acc = optimal_results_subset['accuracy'].iloc[0] if len(optimal_results_subset) > 0 else current_acc
        recommended_acc = recommended_results_subset['accuracy'].iloc[0] if len(recommended_results_subset) > 0 else current_acc
        
        optimal_threshold = optimal_results_subset['threshold'].iloc[0] if len(optimal_results_subset) > 0 else 0.5
        recommended_threshold = recommended_results_subset['threshold'].iloc[0] if len(recommended_results_subset) > 0 else 0.5
        
        summary_results.append({
            'dataset': dataset,
            'current_accuracy_50pct': current_acc,
            'optimal_threshold': optimal_threshold,
            'optimal_accuracy': optimal_acc,
            'optimal_improvement': optimal_acc - current_acc,
            'recommended_threshold': recommended_threshold,
            'recommended_accuracy': recommended_acc,
            'recommended_improvement': recommended_acc - current_acc,
            'best_strategy': 'optimal' if optimal_acc >= recommended_acc else 'recommended'
        })
    
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(f'{output_dir}/ai_model_threshold_summary.csv', index=False, encoding='utf-8-sig')
    
    # 生成文本报告
    with open(f'{output_dir}/ai_model_threshold_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== AI模型阈值优化分析报告 ===\n\n")
        
        f.write("## 当前评估方法说明\n")
        f.write("当前脚本使用固定的50%阈值对所有数据集进行二分类：\n")
        f.write("- 风险预测值 >= 50% → 预测为高风险(正类)\n")
        f.write("- 风险预测值 < 50% → 预测为低风险(负类)\n\n")
        
        f.write("## 问题分析\n")
        f.write("使用统一的50%阈值存在以下问题：\n")
        f.write("1. 不同数据集的风险分布差异很大\n")
        f.write("2. 每个数据集都有其特定的最优阈值\n")
        f.write("3. 固定阈值无法充分发挥AI模型的预测能力\n\n")
        
        f.write("## 优化建议\n")
        for idx, row in summary_df.iterrows():
            f.write(f"\n### {row['dataset']}\n")
            f.write(f"- 当前准确率(50%阈值): {row['current_accuracy_50pct']:.4f}\n")
            f.write(f"- 最优阈值: {row['optimal_threshold']:.3f}, 准确率: {row['optimal_accuracy']:.4f} (提升: {row['optimal_improvement']:+.4f})\n")
            f.write(f"- 推荐阈值: {row['recommended_threshold']:.3f}, 准确率: {row['recommended_accuracy']:.4f} (提升: {row['recommended_improvement']:+.4f})\n")
            f.write(f"- 建议使用: {'最优阈值' if row['best_strategy'] == 'optimal' else '推荐阈值'}\n")
        
        # 总体改进潜力
        total_current = summary_df['current_accuracy_50pct'].mean()
        total_optimal = summary_df['optimal_accuracy'].mean()
        total_recommended = summary_df['recommended_accuracy'].mean()
        
        f.write(f"\n## 总体改进潜力\n")
        f.write(f"- 当前平均准确率: {total_current:.4f}\n")
        f.write(f"- 最优阈值平均准确率: {total_optimal:.4f} (提升: {total_optimal - total_current:+.4f})\n")
        f.write(f"- 推荐阈值平均准确率: {total_recommended:.4f} (提升: {total_recommended - total_current:+.4f})\n")
    
    print("✓ 综合分析报告已生成")
    
    return detailed_df, summary_df

def main():
    """主函数"""
    # 创建输出目录
    output_dir = 'results/rq2_performance_analysis'
    
    # 加载数据
    patient_data = load_patient_data()
    
    # 解释当前评估方法
    explain_current_evaluation_method()
    
    # 寻找最优阈值
    optimal_results = find_optimal_thresholds(patient_data)
    
    # 分析推荐阈值
    recommended_results = analyze_dataset_specific_thresholds(patient_data)
    
    # 生成详细表格
    if optimal_results:
        create_threshold_analysis_tables(optimal_results, output_dir)
    
    # 生成综合报告
    detailed_df, summary_df = generate_comprehensive_report(
        optimal_results, recommended_results, patient_data, output_dir)
    
    # 显示关键发现
    print("\n🎯 关键发现:")
    print("=" * 50)
    
    for idx, row in summary_df.iterrows():
        dataset = row['dataset']
        current_acc = row['current_accuracy_50pct']
        optimal_acc = row['optimal_accuracy']
        optimal_threshold = row['optimal_threshold']
        improvement = row['optimal_improvement']
        
        print(f"\n📊 {dataset}:")
        print(f"   当前准确率(50%): {current_acc:.1%}")
        print(f"   最优准确率({optimal_threshold:.1%}): {optimal_acc:.1%}")
        print(f"   潜在提升: {improvement:+.1%}")
    
    total_improvement = summary_df['optimal_improvement'].mean()
    print(f"\n🚀 总体潜在提升: {total_improvement:+.1%}")
    
    print(f"\n📁 所有分析文件已保存到: {output_dir}/")
    print("   - ai_model_threshold_analysis.csv (详细结果)")
    print("   - ai_model_threshold_summary.csv (汇总结果)")
    print("   - ai_model_threshold_report.txt (分析报告)")
    print("   - threshold_details_all.csv (所有阈值详细性能)")

if __name__ == "__main__":
    main()
