#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取SUS系统易用性量表数据
"""

import pandas as pd
import json
import numpy as np

def calculate_sus_scores(df):
    """计算SUS分数"""
    
    # SUS评分映射（5点Likert量表）
    likert_mapping = {
        '非常同意': 5,
        '同意': 4,
        '中立': 3,
        '不同意': 2,
        '非常不同意': 1,
        '(空)': np.nan  # 处理空值
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
    
    # 添加百分位排名
    def get_sus_percentile(score):
        if pd.isna(score):
            return 'N/A'
        elif score >= 85:
            return '90+'
        elif score >= 80:
            return '80-90'
        elif score >= 70:
            return '70-80'
        elif score >= 60:
            return '50-70'
        elif score >= 50:
            return '30-50'
        else:
            return '<30'
    
    result_df['SUS_Grade'] = result_df['SUS_Score'].apply(get_sus_grade)
    result_df['SUS_Percentile'] = result_df['SUS_Score'].apply(get_sus_percentile)
    
    return result_df

def calculate_question_statistics(df):
    """计算每个SUS问题的统计信息"""
    
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
        return pd.DataFrame()
    
    # 创建统计结果DataFrame
    stats_data = []
    
    # 计算每个问题的统计信息
    for col in sus_columns:
        # 转换为数值
        numeric_values = df[col].map(likert_mapping)
        valid_values = numeric_values.dropna()
        
        if len(valid_values) > 0:
            stats_data.append({
                '问题编号': col,
                '问题描述': get_question_description(col),
                '有效回答数': len(valid_values),
                '平均值': valid_values.mean(),
                '方差': valid_values.var(),
                '标准差': valid_values.std(),
                '最小值': valid_values.min(),
                '最大值': valid_values.max(),
                '中位数': valid_values.median()
            })
    
    # 添加总体SUS分数统计
    if 'SUS_Score' in df.columns:
        sus_scores = df['SUS_Score'].dropna()
        if len(sus_scores) > 0:
            stats_data.append({
                '问题编号': 'SUS_总分',
                '问题描述': 'SUS总体可用性分数 (0-100分)',
                '有效回答数': len(sus_scores),
                '平均值': sus_scores.mean(),
                '方差': sus_scores.var(),
                '标准差': sus_scores.std(),
                '最小值': sus_scores.min(),
                '最大值': sus_scores.max(),
                '中位数': sus_scores.median()
            })
    
    return pd.DataFrame(stats_data)

def get_question_description(col_name):
    """获取问题的中文描述"""
    descriptions = {
        'SUS_Q1_经常使用': '我想我会愿意在临床工作中经常使用这个系统',
        'SUS_Q2_复杂性_R': '我发现这个系统没必要地复杂 (反向)',
        'SUS_Q3_容易上手': '我觉得这个系统很容易上手',
        'SUS_Q4_需要支持_R': '我想我需要技术人员的支持才能使用这个系统 (反向)',
        'SUS_Q5_功能整合': '我发现系统里的各项功能都很好地整合在了一起',
        'SUS_Q6_设计矛盾_R': '我觉得这个系统在设计上存在矛盾或不一致的地方 (反向)',
        'SUS_Q7_快速学会': '我想大多数医生都能很快学会如何使用这个系统',
        'SUS_Q8_笨重繁琐_R': '我感觉这个系统用起来非常笨重和繁琐 (反向)',
        'SUS_Q9_安心自信': '我使用这个系统时感到很安心和自信',
        'SUS_Q10_学习时间_R': '在使用这个系统之前，我需要花很多时间学习 (反向)'
    }
    return descriptions.get(col_name, col_name)

def calculate_department_statistics(df):
    """计算按科室的SUS统计信息"""
    
    if '科室' not in df.columns:
        return pd.DataFrame()
    
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
    
    dept_stats = []
    
    # 按科室分组统计
    for dept in df['科室'].unique():
        dept_data = df[df['科室'] == dept]
        
        # 统计总体SUS分数
        if 'SUS_Score' in dept_data.columns:
            sus_scores = dept_data['SUS_Score'].dropna()
            if len(sus_scores) > 0:
                dept_stats.append({
                    '科室': dept,
                    '指标': 'SUS_总分',
                    '指标描述': 'SUS总体可用性分数 (0-100分)',
                    '参与者数': len(sus_scores),
                    '平均值': sus_scores.mean(),
                    '方差': sus_scores.var(),
                    '标准差': sus_scores.std(),
                    '最小值': sus_scores.min(),
                    '最大值': sus_scores.max(),
                    '中位数': sus_scores.median()
                })
        
        # 统计每个SUS问题
        for col in sus_columns:
            if col in dept_data.columns:
                # 转换为数值
                numeric_values = dept_data[col].map(likert_mapping)
                valid_values = numeric_values.dropna()
                
                if len(valid_values) > 0:
                    dept_stats.append({
                        '科室': dept,
                        '指标': col,
                        '指标描述': get_question_description(col),
                        '参与者数': len(valid_values),
                        '平均值': valid_values.mean(),
                        '方差': valid_values.var(),
                        '标准差': valid_values.std(),
                        '最小值': valid_values.min(),
                        '最大值': valid_values.max(),
                        '中位数': valid_values.median()
                    })
    
    return pd.DataFrame(dept_stats)

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
    
    # 创建重命名映射（只对存在的列）
    sus_col_index = 0
    for col in sus_cols:
        if col in df.columns:
            sus_rename_map[col] = sus_question_names[sus_col_index]
        sus_col_index += 1
    
    # 应用重命名
    sus_data = sus_data.rename(columns=sus_rename_map)
    
    # 计算SUS分数
    sus_data_with_scores = calculate_sus_scores(sus_data.copy())
    
    # 计算每个SUS问题的统计信息
    sus_question_stats = calculate_question_statistics(sus_data_with_scores)
    
    # 保存原始数据和计算结果
    output_file = 'results/sus_raw_data.csv'
    sus_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"SUS原始数据已保存到: {output_file}")
    
    # 保存包含分数的数据
    scores_file = 'results/sus_analysis.csv'
    sus_data_with_scores.to_csv(scores_file, index=False, encoding='utf-8-sig')
    print(f"SUS分析结果已保存到: {scores_file}")
    
    # 保存问题统计数据
    stats_file = 'results/sus_question_statistics.csv'
    sus_question_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"SUS问题统计数据已保存到: {stats_file}")
    
    # 计算并保存按科室的统计
    dept_stats = calculate_department_statistics(sus_data_with_scores)
    if not dept_stats.empty:
        dept_stats_file = 'results/sus_department_statistics.csv'
        dept_stats.to_csv(dept_stats_file, index=False, encoding='utf-8-sig')
        print(f"按科室SUS统计数据已保存到: {dept_stats_file}")
    
    # 显示数据概览
    print(f"\n数据概览:")
    print(f"参与者数量: {len(sus_data)}")
    print(f"数据列数: {len(sus_data.columns)}")
    
    # 显示SUS分数统计
    if 'SUS_Score' in sus_data_with_scores.columns:
        valid_scores = sus_data_with_scores['SUS_Score'].dropna()
        if len(valid_scores) > 0:
            print(f"\nSUS分数统计:")
            print(f"平均分: {valid_scores.mean():.1f}")
            print(f"中位数: {valid_scores.median():.1f}")
            print(f"标准差: {valid_scores.std():.1f}")
            print(f"最高分: {valid_scores.max():.1f}")
            print(f"最低分: {valid_scores.min():.1f}")
            
            # 按等级分类
            print(f"\nSUS分数等级分布:")
            for _, row in sus_data_with_scores.iterrows():
                if not pd.isna(row['SUS_Score']):
                    score = row['SUS_Score']
                    grade = row['SUS_Grade']
                    percentile = row['SUS_Percentile']
                    print(f"  {row['ID']}: {score:.1f}分 ({grade}, {percentile}%分位)")
    
    # 显示按科室的SUS分数
    if '科室' in sus_data_with_scores.columns and 'SUS_Score' in sus_data_with_scores.columns:
        print(f"\n按科室的SUS分数:")
        dept_scores = sus_data_with_scores.groupby('科室')['SUS_Score'].agg(['mean', 'std', 'count'])
        print(dept_scores)
    
    # 显示每个问题的统计信息
    if not sus_question_stats.empty:
        print(f"\n=== SUS问题详细统计 ===")
        for _, row in sus_question_stats.iterrows():
            print(f"\n{row['问题编号']}: {row['问题描述']}")
            print(f"  有效回答数: {row['有效回答数']}")
            print(f"  平均值: {row['平均值']:.2f}")
            print(f"  方差: {row['方差']:.2f}")
            print(f"  标准差: {row['标准差']:.2f}")
            print(f"  范围: {row['最小值']:.1f} - {row['最大值']:.1f}")
            print(f"  中位数: {row['中位数']:.2f}")
    
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