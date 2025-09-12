#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取SUS系统易用性量表数据 - 简化版
"""

import pandas as pd
import numpy as np
import json

def calculate_sus_scores(df):
    """计算SUS分数"""
    
    # SUS评分映射（5点Likert量表）
    likert_mapping = {
        '非常同意': 5,
        '同意': 4,
        '中立': 3,
        '不同意': 2,
        '非常不同意': 1,
        '(空)': np.nan
    }
    
    # 获取SUS相关列
    sus_columns = [col for col in df.columns if col.startswith('SUS_Q')]
    
    if len(sus_columns) == 0:
        print("警告：未找到SUS问题列")
        return df
    
    # 转换为数值
    sus_numeric = df[sus_columns].copy()
    for col in sus_columns:
        sus_numeric[col] = sus_numeric[col].map(likert_mapping)
    
    # 反向计分题目（题目2, 4, 6, 8, 10）
    reverse_items = ['SUS_Q2_复杂性_R', 'SUS_Q4_需要支持_R', 'SUS_Q6_设计矛盾_R', 'SUS_Q8_笨重繁琐_R', 'SUS_Q10_学习时间_R']
    
    for col in reverse_items:
        if col in sus_numeric.columns:
            sus_numeric[col] = 6 - sus_numeric[col]  # 5->1, 4->2, 3->3, 2->4, 1->5
    
    # 计算SUS分数
    sus_scores = []
    for idx, row in sus_numeric.iterrows():
        valid_scores = [score for score in row[sus_columns] if not pd.isna(score)]
        if len(valid_scores) >= 8:  # 至少需要8个有效回答
            total_score = sum(valid_scores)
            # 标准SUS计算：(总分 - 题目数) * 2.5
            sus_score = (total_score - len(valid_scores)) * 2.5
            sus_scores.append(sus_score)
        else:
            sus_scores.append(np.nan)
    
    # 添加分数到数据框
    result_df = df.copy()
    result_df['SUS_Score'] = sus_scores
    
    # 添加等级评价
    def get_sus_grade(score):
        if pd.isna(score):
            return 'N/A'
        elif score >= 80:
            return 'A (优秀)'
        elif score >= 68:
            return 'B (良好)'
        elif score >= 51:
            return 'C (一般)'
        else:
            return 'D (差)'
    
    result_df['SUS_Grade'] = result_df['SUS_Score'].apply(get_sus_grade)
    
    return result_df

def create_sus_statistics(df):
    """创建SUS统计分析表"""
    
    # SUS评分映射
    likert_mapping = {
        '非常同意': 5,
        '同意': 4,
        '中立': 3,
        '不同意': 2,
        '非常不同意': 1,
        '(空)': np.nan
    }
    
    # 问题描述映射
    question_descriptions = {
        'SUS_Q1_经常使用': 'Q1: 我想我会愿意在临床工作中经常使用这个系统',
        'SUS_Q2_复杂性_R': 'Q2: 我发现这个系统没必要地复杂 (反向)',
        'SUS_Q3_容易上手': 'Q3: 我觉得这个系统很容易上手',
        'SUS_Q4_需要支持_R': 'Q4: 我想我需要技术人员的支持才能使用这个系统 (反向)',
        'SUS_Q5_功能整合': 'Q5: 我发现系统里的各项功能都很好地整合在了一起',
        'SUS_Q6_设计矛盾_R': 'Q6: 我觉得这个系统在设计上存在矛盾或不一致的地方 (反向)',
        'SUS_Q7_快速学会': 'Q7: 我想大多数医生都能很快学会如何使用这个系统',
        'SUS_Q8_笨重繁琐_R': 'Q8: 我感觉这个系统用起来非常笨重和繁琐 (反向)',
        'SUS_Q9_安心自信': 'Q9: 我使用这个系统时感到很安心和自信',
        'SUS_Q10_学习时间_R': 'Q10: 在使用这个系统之前，我需要花很多时间学习 (反向)'
    }
    
    stats_data = []
    
    # 1. 总体SUS分数统计
    if 'SUS_Score' in df.columns:
        sus_scores = df['SUS_Score'].dropna()
        if len(sus_scores) > 0:
            stats_data.append({
                '指标类型': '总体分数',
                '指标名称': 'SUS总分',
                '指标描述': 'SUS系统易用性总分 (0-100分)',
                '样本数': len(sus_scores),
                '平均值': round(sus_scores.mean(), 2),
                '方差': round(sus_scores.var(), 2),
                '标准差': round(sus_scores.std(), 2),
                '最小值': round(sus_scores.min(), 2),
                '最大值': round(sus_scores.max(), 2),
                '中位数': round(sus_scores.median(), 2)
            })
    
    # 2. 各问题统计（转换为百分制）
    sus_columns = [col for col in df.columns if col.startswith('SUS_Q')]
    for col in sus_columns:
        numeric_values = df[col].map(likert_mapping)
        valid_values = numeric_values.dropna()
        
        if len(valid_values) > 0:
            # 转换为百分制：(分数-1) * 25，使1-5分对应0-100分
            percentage_values = (valid_values - 1) * 25
            
            stats_data.append({
                '指标类型': '单项问题',
                '指标名称': col,
                '指标描述': question_descriptions.get(col, col),
                '样本数': len(percentage_values),
                '平均值': round(percentage_values.mean(), 2),
                '方差': round(percentage_values.var(), 2),
                '标准差': round(percentage_values.std(), 2),
                '最小值': round(percentage_values.min(), 2),
                '最大值': round(percentage_values.max(), 2),
                '中位数': round(percentage_values.median(), 2)
            })
    
    # 3. 按科室统计
    if '科室' in df.columns and 'SUS_Score' in df.columns:
        for dept in df['科室'].unique():
            dept_data = df[df['科室'] == dept]
            dept_scores = dept_data['SUS_Score'].dropna()
            
            if len(dept_scores) > 0:
                stats_data.append({
                    '指标类型': '科室分析',
                    '指标名称': f'{dept}_SUS总分',
                    '指标描述': f'{dept}科室SUS总分',
                    '样本数': len(dept_scores),
                    '平均值': round(dept_scores.mean(), 2),
                    '方差': round(dept_scores.var(), 2),
                    '标准差': round(dept_scores.std(), 2),
                    '最小值': round(dept_scores.min(), 2),
                    '最大值': round(dept_scores.max(), 2),
                    '中位数': round(dept_scores.median(), 2)
                })
    
    # 4. 按年资统计
    if '年资分类' in df.columns and 'SUS_Score' in df.columns:
        for seniority in df['年资分类'].unique():
            seniority_data = df[df['年资分类'] == seniority]
            seniority_scores = seniority_data['SUS_Score'].dropna()
            
            if len(seniority_scores) > 0:
                stats_data.append({
                    '指标类型': '年资分析',
                    '指标名称': f'{seniority}_SUS总分',
                    '指标描述': f'{seniority}医护人员SUS总分',
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
    """提取SUS数据"""
    
    # 读取合并后的数据
    print("正在读取数据文件...")
    df = pd.read_excel('results/datasets/merged_dataset_simple.xlsx')
    
    # 读取年资分类数据
    print("正在加载年资分类...")
    try:
        with open('results/participants/seniority_classification.json', 'r', encoding='utf-8') as f:
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
    
    # 定义SUS量表的10个题目列
    sus_cols = [
        '72、系统易用性量表（SUS）针对“小雅医生”系统，请根据您的使用体验，选择最符合您看法的选项。—我想我会愿意在临床工作中经常使用这个系统。',
        '72、我发现这个系统没必要地复杂。',
        '72、我觉得这个系统很容易上手。',
        '72、我想我需要技术人员的支持才能使用这个系统。',
        '72、我发现系统里的各项功能（如图表、列表）都很好地整合在了一起。',
        '72、我觉得这个系统在设计上存在矛盾或不一致的地方。',
        '72、我想大多数医生都能很快学会如何使用这个系统。',
        '72、我感觉这个系统用起来非常笨重和繁琐。',
        '72、我使用这个系统时感到很安心和自信。',
        '72、在使用这个系统之前，我需要花很多时间学习。'
    ]
    
    # 检查列是否存在
    available_cols = [col for col in participant_cols + sus_cols if col in df.columns]
    sus_data = df[available_cols].copy()
    
    # 重命名SUS题目列为简洁的名称
    sus_rename_map = {}
    sus_question_names = [
        'SUS_Q1_经常使用',
        'SUS_Q2_复杂性_R',
        'SUS_Q3_容易上手',
        'SUS_Q4_需要支持_R',
        'SUS_Q5_功能整合',
        'SUS_Q6_设计矛盾_R',
        'SUS_Q7_快速学会',
        'SUS_Q8_笨重繁琐_R',
        'SUS_Q9_安心自信',
        'SUS_Q10_学习时间_R'
    ]
    
    sus_col_index = 0
    for col in sus_cols:
        if col in df.columns:
            sus_rename_map[col] = sus_question_names[sus_col_index]
        sus_col_index += 1
    
    sus_data = sus_data.rename(columns=sus_rename_map)
    
    # 计算SUS分数
    sus_data_with_scores = calculate_sus_scores(sus_data.copy())
    
    # 创建统计分析表
    sus_statistics = create_sus_statistics(sus_data_with_scores)
    
    # 保存两个核心文件
    
    # 1. 原始数据文件（包含所有原始信息和计算的分数）
    raw_file = 'results/sus/sus_raw_data.csv'
    sus_data_with_scores.to_csv(raw_file, index=False, encoding='utf-8-sig')
    print(f"✓ SUS原始数据（含分数）已保存到: {raw_file}")
    
    # 2. 统计分析文件（关键统计信息）
    stats_file = 'results/sus/sus_analysis.csv'
    sus_statistics.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"✓ SUS统计分析已保存到: {stats_file}")
    
    # 显示关键结果
    print(f"\n=== SUS分析结果概览 ===")
    print(f"参与者数量: {len(sus_data)}")
    
    if 'SUS_Score' in sus_data_with_scores.columns:
        valid_scores = sus_data_with_scores['SUS_Score'].dropna()
        if len(valid_scores) > 0:
            print(f"SUS平均分: {valid_scores.mean():.1f}分")
            print(f"分数范围: {valid_scores.min():.1f} - {valid_scores.max():.1f}分")
            
            # 按科室显示
            if '科室' in sus_data_with_scores.columns:
                print(f"\n按科室SUS分数:")
                dept_scores = sus_data_with_scores.groupby('科室')['SUS_Score'].agg(['count', 'mean', 'std']).round(1)
                dept_scores.columns = ['人数', '平均分', '标准差']
                print(dept_scores)
            
            # 按年资显示
            if '年资分类' in sus_data_with_scores.columns:
                print(f"\n按年资SUS分数:")
                seniority_scores = sus_data_with_scores.groupby('年资分类')['SUS_Score'].agg(['count', 'mean', 'std']).round(1)
                seniority_scores.columns = ['人数', '平均分', '标准差']
                print(seniority_scores)

if __name__ == "__main__":
    main()