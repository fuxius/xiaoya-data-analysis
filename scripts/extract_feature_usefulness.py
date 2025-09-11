#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取AICare系统功能有用性评价数据 - 简化版
"""

import pandas as pd
import numpy as np
import json

def calculate_usefulness_scores(df):
    """计算功能有用性分数"""
    
    # 有用性评分映射（5点Likert量表）
    usefulness_mapping = {
        '极其有用': 4,
        '比较有用': 3,
        '一般有用': 2,
        '有点没用': 1,
        '完全没用': 0,
        '(空)': np.nan
    }
    
    # 获取功能评价相关列
    feature_columns = [col for col in df.columns if col.startswith('F')]
    
    if len(feature_columns) == 0:
        print("警告：未找到功能评价列")
        return df
    
    # 转换为数值
    feature_numeric = df[feature_columns].copy()
    for col in feature_columns:
        feature_numeric[col] = feature_numeric[col].map(usefulness_mapping)
    
    # 计算每个功能的平均有用性分数并换算成百分制
    usefulness_scores = []
    for idx, row in feature_numeric.iterrows():
        valid_scores = [score for score in row[feature_columns] if not pd.isna(score)]
        if len(valid_scores) >= 3:  # 至少需要3个有效回答
            avg_score = np.mean(valid_scores)
            # 换算成百分制：(平均分 / 4) * 100
            usefulness_score_100 = (avg_score / 4.0) * 100
            usefulness_scores.append(usefulness_score_100)
        else:
            usefulness_scores.append(np.nan)
    
    # 添加分数到数据框
    result_df = df.copy()
    result_df['Usefulness_Score'] = usefulness_scores
    
    # 添加等级评价
    def get_usefulness_grade(score):
        if pd.isna(score):
            return 'N/A'
        elif score >= 87.5:  # 3.5/4 * 100
            return 'A (极其有用)'
        elif score >= 62.5:  # 2.5/4 * 100
            return 'B (比较有用)'
        elif score >= 37.5:  # 1.5/4 * 100
            return 'C (一般有用)'
        elif score >= 12.5:  # 0.5/4 * 100
            return 'D (有点没用)'
        else:
            return 'E (完全没用)'
    
    result_df['Usefulness_Grade'] = result_df['Usefulness_Score'].apply(get_usefulness_grade)
    
    return result_df

def create_usefulness_statistics(df):
    """创建功能有用性统计分析表"""
    
    # 有用性评分映射
    usefulness_mapping = {
        '极其有用': 4,
        '比较有用': 3,
        '一般有用': 2,
        '有点没用': 1,
        '完全没用': 0,
        '(空)': np.nan
    }
    
    # 功能描述映射
    feature_descriptions = {
        'F1_动态风险轨迹可视化': 'F1: 动态风险轨迹可视化 - 显示患者每次就诊时风险预测的折线图',
        'F2_个体化关键指标分析': 'F2: 个体化关键指标分析 - 每次就诊的特征重要性分析，支持交互式探索',
        'F3_人群级别指标分析': 'F3: 人群级别指标分析 - 数据集中特征重要性、数值和患者风险的3D和2D可视化',
        'F4_LLM诊疗建议': 'F4: LLM诊疗建议 - 基于患者EHR数据、风险轨迹和临床指标提供分析和建议'
    }
    
    stats_data = []
    
    # 1. 总体有用性分数统计
    if 'Usefulness_Score' in df.columns:
        usefulness_scores = df['Usefulness_Score'].dropna()
        if len(usefulness_scores) > 0:
            stats_data.append({
                '指标类型': '总体分数',
                '指标名称': '功能有用性总分',
                '指标描述': '四个功能模块平均有用性分数 (0-100分)',
                '样本数': len(usefulness_scores),
                '平均值': round(usefulness_scores.mean(), 2),
                '方差': round(usefulness_scores.var(), 2),
                '标准差': round(usefulness_scores.std(), 2),
                '最小值': round(usefulness_scores.min(), 2),
                '最大值': round(usefulness_scores.max(), 2),
                '中位数': round(usefulness_scores.median(), 2)
            })
    
    # 2. 各功能统计（换算成百分制）
    feature_columns = [col for col in df.columns if col.startswith('F')]
    for col in feature_columns:
        numeric_values = df[col].map(usefulness_mapping)
        valid_values = numeric_values.dropna()
        
        if len(valid_values) > 0:
            # 换算成百分制
            valid_values_100 = (valid_values / 4.0) * 100
            stats_data.append({
                '指标类型': '单项功能',
                '指标名称': col,
                '指标描述': feature_descriptions.get(col, col),
                '样本数': len(valid_values_100),
                '平均值': round(valid_values_100.mean(), 2),
                '方差': round(valid_values_100.var(), 2),
                '标准差': round(valid_values_100.std(), 2),
                '最小值': round(valid_values_100.min(), 2),
                '最大值': round(valid_values_100.max(), 2),
                '中位数': round(valid_values_100.median(), 2)
            })
    
    # 3. 按科室统计
    if '科室' in df.columns and 'Usefulness_Score' in df.columns:
        for dept in df['科室'].unique():
            dept_data = df[df['科室'] == dept]
            dept_scores = dept_data['Usefulness_Score'].dropna()
            
            if len(dept_scores) > 0:
                stats_data.append({
                    '指标类型': '科室分析',
                    '指标名称': f'{dept}_功能有用性总分',
                    '指标描述': f'{dept}科室功能有用性总分',
                    '样本数': len(dept_scores),
                    '平均值': round(dept_scores.mean(), 2),
                    '方差': round(dept_scores.var(), 2),
                    '标准差': round(dept_scores.std(), 2),
                    '最小值': round(dept_scores.min(), 2),
                    '最大值': round(dept_scores.max(), 2),
                    '中位数': round(dept_scores.median(), 2)
                })
    
    # 4. 按年资统计
    if '年资分类' in df.columns and 'Usefulness_Score' in df.columns:
        for seniority in df['年资分类'].unique():
            seniority_data = df[df['年资分类'] == seniority]
            seniority_scores = seniority_data['Usefulness_Score'].dropna()
            
            if len(seniority_scores) > 0:
                stats_data.append({
                    '指标类型': '年资分析',
                    '指标名称': f'{seniority}_功能有用性总分',
                    '指标描述': f'{seniority}医护人员功能有用性总分',
                    '样本数': len(seniority_scores),
                    '平均值': round(seniority_scores.mean(), 2),
                    '方差': round(seniority_scores.var(), 2),
                    '标准差': round(seniority_scores.std(), 2),
                    '最小值': round(seniority_scores.min(), 2),
                    '最大值': round(seniority_scores.max(), 2),
                    '中位数': round(seniority_scores.median(), 2)
                })
    
    return pd.DataFrame(stats_data)

def main():
    """提取功能有用性数据"""
    
    # 读取合并后的数据
    print("正在读取数据文件...")
    df = pd.read_excel('results/merged_dataset_simple.xlsx')
    
    # 读取年资分类数据
    print("正在加载年资分类...")
    try:
        with open('results/seniority_classification.json', 'r', encoding='utf-8') as f:
            seniority_data = json.load(f)
        
        # 添加年资分类到数据框
        df['年资分类'] = df['ID'].map(lambda x: seniority_data['participant_classification'].get(x, {}).get('seniority_level', '未知'))
        print(f"✓ 年资分类已加载")
    except FileNotFoundError:
        print("⚠️ 未找到年资分类文件，将不进行年资分析")
        df['年资分类'] = '未知'
    
    # 定义参与者基本信息列
    participant_cols = [
        'ID',
        '科室',
        '7、您当前的身份或职称是？',
        '年资分类'
    ]
    
    # 定义四个核心功能模块的评价列
    feature_cols = [
        '74、“小雅医生”系统功能反馈请评价“小雅医生”系统的各项功能对您的诊断过程有多大帮助。—动态风险轨迹可视化',
        '74、可交互的个体化关键指标列表',
        '74、人群层面的指标分析可视化',
        '74、大语言模型驱动的诊疗建议'
    ]
    
    # 检查列是否存在
    available_cols = [col for col in participant_cols + feature_cols if col in df.columns]
    feature_data = df[available_cols].copy()
    
    # 重命名功能评价列为简洁的名称
    feature_rename_map = {
        '74、“小雅医生”系统功能反馈请评价“小雅医生”系统的各项功能对您的诊断过程有多大帮助。—动态风险轨迹可视化': 'F1_动态风险轨迹可视化',
        '74、可交互的个体化关键指标列表': 'F2_个体化关键指标分析',
        '74、人群层面的指标分析可视化': 'F3_人群级别指标分析',
        '74、大语言模型驱动的诊疗建议': 'F4_LLM诊疗建议'
    }
    
    feature_data = feature_data.rename(columns=feature_rename_map)
    
    # 计算有用性分数
    feature_data_with_scores = calculate_usefulness_scores(feature_data.copy())
    
    # 创建统计分析表
    feature_statistics = create_usefulness_statistics(feature_data_with_scores)
    
    # 保存两个核心文件
    
    # 1. 原始数据文件（包含所有原始信息和计算的分数）
    raw_file = 'results/feature_usefulness_raw_data.csv'
    feature_data_with_scores.to_csv(raw_file, index=False, encoding='utf-8-sig')
    print(f"✓ 功能有用性原始数据（含分数）已保存到: {raw_file}")
    
    # 2. 统计分析文件（关键统计信息）
    stats_file = 'results/feature_usefulness_analysis.csv'
    feature_statistics.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"✓ 功能有用性统计分析已保存到: {stats_file}")
    
    # 显示关键结果
    print(f"\n=== 功能有用性分析结果概览 ===")
    print(f"参与者数量: {len(feature_data)}")
    
    if 'Usefulness_Score' in feature_data_with_scores.columns:
        valid_scores = feature_data_with_scores['Usefulness_Score'].dropna()
        if len(valid_scores) > 0:
            print(f"功能有用性平均分: {valid_scores.mean():.1f}分 (0-100分制)")
            print(f"分数范围: {valid_scores.min():.1f} - {valid_scores.max():.1f}分")
            
            # 按科室显示
            if '科室' in feature_data_with_scores.columns:
                print(f"\n按科室功能有用性分数:")
                dept_scores = feature_data_with_scores.groupby('科室')['Usefulness_Score'].agg(['count', 'mean', 'std']).round(1)
                dept_scores.columns = ['人数', '平均分', '标准差']
                print(dept_scores)
            
            # 按年资显示
            if '年资分类' in feature_data_with_scores.columns:
                print(f"\n按年资功能有用性分数:")
                seniority_scores = feature_data_with_scores.groupby('年资分类')['Usefulness_Score'].agg(['count', 'mean', 'std']).round(1)
                seniority_scores.columns = ['人数', '平均分', '标准差']
                print(seniority_scores)
    
    # 显示各功能模块的详细统计
    feature_columns = [col for col in feature_data_with_scores.columns if col.startswith('F')]
    if feature_columns:
        print(f"\n各功能模块平均分 (0-100分制):")
        usefulness_mapping = {
            '极其有用': 4,
            '比较有用': 3,
            '一般有用': 2,
            '有点没用': 1,
            '完全没用': 0,
            '(空)': np.nan
        }
        
        for col in feature_columns:
            if col in feature_data_with_scores.columns:
                numeric_values = feature_data_with_scores[col].map(usefulness_mapping)
                valid_values = numeric_values.dropna()
                if len(valid_values) > 0:
                    # 换算成百分制
                    score_100 = (valid_values.mean() / 4.0) * 100
                    print(f"  {col}: {score_100:.1f}分")

if __name__ == "__main__":
    main()