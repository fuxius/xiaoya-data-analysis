#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取SUS系统易用性量表数据
"""

import pandas as pd
import json

def main():
    """提取SUS数据"""
    
    # 读取合并后的数据
    print("正在读取数据文件...")
    df = pd.read_excel('results/merged_dataset_simple.xlsx')
    
    # 定义参与者基本信息列
    participant_cols = [
        'ID',
        '科室',
        '7、您当前的身份或职称是？'
    ]
    
    # 定义SUS量表的10个题目列（根据标准SUS量表）
    sus_cols = [
        '72、系统易用性量表（SUS）针对"小雅医生"系统，请根据您的使用体验，选择最符合您看法的选项。—我想我会愿意在临床工作中经常使用这个系统。',
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
    missing_cols = []
    for col in participant_cols + sus_cols:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"警告：以下列不存在于数据中：")
        for col in missing_cols:
            print(f"  - {col}")
    
    # 提取存在的列
    available_cols = [col for col in participant_cols + sus_cols if col in df.columns]
    sus_data = df[available_cols].copy()
    
    # 重命名SUS题目列为简洁的名称
    sus_rename_map = {}
    sus_question_names = [
        'SUS_Q1_经常使用',
        'SUS_Q2_复杂性_R',  # R表示反向计分
        'SUS_Q3_容易上手',
        'SUS_Q4_需要支持_R',
        'SUS_Q5_功能整合',
        'SUS_Q6_设计矛盾_R',
        'SUS_Q7_快速学会',
        'SUS_Q8_笨重繁琐_R',
        'SUS_Q9_安心自信',
        'SUS_Q10_学习时间_R'
    ]
    
    # 创建重命名映射
    sus_col_index = 0
    for col in sus_cols:
        if col in df.columns:
            sus_rename_map[col] = sus_question_names[sus_col_index]
        sus_col_index += 1
    
    # 应用重命名
    sus_data = sus_data.rename(columns=sus_rename_map)
    
    # 计算SUS分数
    sus_data_with_scores = calculate_sus_scores(sus_data.copy())
    
    # 保存原始数据和计算结果
    output_file = 'results/sus_raw_data.csv'
    sus_data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"SUS原始数据已保存到: {output_file}")
    
    # 保存包含分数的数据
    scores_file = 'results/sus_analysis.csv'
    sus_data_with_scores.to_csv(scores_file, index=False, encoding='utf-8')
    print(f"SUS分析结果已保存到: {scores_file}")
    
    # 显示数据概览
    print(f"\n数据概览:")
    print(f"参与者数量: {len(sus_data)}")
    print(f"数据列数: {len(sus_data.columns)}")
    
    # 显示SUS分数统计
    if 'SUS_Score' in sus_data_with_scores.columns:
        print(f"\nSUS分数统计:")
        print(f"平均分: {sus_data_with_scores['SUS_Score'].mean():.1f}")
        print(f"中位数: {sus_data_with_scores['SUS_Score'].median():.1f}")
        print(f"标准差: {sus_data_with_scores['SUS_Score'].std():.1f}")
        print(f"最高分: {sus_data_with_scores['SUS_Score'].max():.1f}")
        print(f"最低分: {sus_data_with_scores['SUS_Score'].min():.1f}")
        
        # 按等级分类
        print(f"\nSUS分数等级分布:")
        for _, row in sus_data_with_scores.iterrows():
            score = row['SUS_Score']
            grade = row['SUS_Grade']
            percentile = row['SUS_Percentile']
            print(f"  {row['ID']}: {score:.1f}分 ({grade}, {percentile}%分位)")
    
    # 显示按科室的SUS分数
    if '科室' in sus_data_with_scores.columns and 'SUS_Score' in sus_data_with_scores.columns:
        print(f"\n按科室的SUS分数:")
        dept_scores = sus_data_with_scores.groupby('科室')['SUS_Score'].agg(['mean', 'std', 'count'])
        print(dept_scores)
    
    print(f"\n参与者基本信息分布:")
    if '科室' in sus_data.columns:
        print("科室分布:")
        print(sus_data['科室'].value_counts())
    
    if '7、您当前的身份或职称是？' in sus_data.columns:
        print("\n职称分布:")
        print(sus_data['7、您当前的身份或职称是？'].value_counts())
    
    # 显示SUS数据示例
    print(f"\nSUS数据示例（前5行）:")
    print(sus_data.head())
    
    # 保存列名映射信息
    mapping_info = {
        'participant_columns': participant_cols,
        'sus_columns': sus_cols,
        'sus_question_mapping': sus_rename_map,
        'reverse_scored_items': ['SUS_Q2_复杂性_R', 'SUS_Q4_需要支持_R', 'SUS_Q6_设计矛盾_R', 'SUS_Q8_笨重繁琐_R', 'SUS_Q10_学习时间_R']
    }
    
    with open('results/sus_column_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n列名映射信息已保存到: results/sus_column_mapping.json")

if __name__ == "__main__":
    main()