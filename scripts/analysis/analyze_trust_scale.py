#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化信任度量表分析（Trust in Automation Scale）
参考NASA-TLX的分析方法，计算信任度得分并比较AI辅助组与无辅助组的信任水平，贯穿不同群体差异分析

作者: AICare研究团队
日期: 2025年9月
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
import re

def load_data_and_create_groups():
    """加载数据并创建分组变量"""
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
    
    # 创建AI使用顺序分组
    df['ai_first'] = df['是否先使用AI分析系统'] == '是'
    
    # 创建医院分组
    df['hospital'] = df['ID'].str[:2]
    
    return df

def clean_trust_score(value):
    """清洗信任度评分数据"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        if value == '(空)':
            return np.nan
        # 定义Likert量表映射
        likert_mapping = {
            '完全同意': 7,
            '部分同意': 6,
            '略微同意': 5,
            '中性': 4,
            '略微不同意': 3,
            '部分不同意': 2,
            '完全不同意': 1
        }
        return likert_mapping.get(value, np.nan)
    return float(value)

def identify_trust_columns(df):
    """识别所有信任度量表相关列"""
    trust_columns = []
    reverse_columns = []
    
    # 查找所有73、开头的列（信任度量表）
    for col in df.columns:
        if col.startswith('73、'):
            trust_columns.append(col)
            # 检查是否是反向计分题（包含 (R)）
            if '(R)' in col:
                reverse_columns.append(col)
    
    print(f"找到信任度量表题目: {len(trust_columns)}个")
    print(f"其中反向计分题: {len(reverse_columns)}个")
    
    return trust_columns, reverse_columns

def calculate_trust_scores(df):
    """计算信任度得分"""
    print("\n正在计算信任度得分...")
    
    # 识别信任度量表列
    trust_columns, reverse_columns = identify_trust_columns(df)
    
    if not trust_columns:
        print("⚠️ 未找到信任度量表数据")
        return pd.DataFrame()
    
    # 为每个参与者计算信任度数据
    trust_data = []
    
    for idx, row in df.iterrows():
        participant_data = {
            'ID': row['ID'],
            '科室': row['科室'],
            '年资分类': row['年资分类'],
            'hospital': row['hospital'],
            'ai_first': row['ai_first']
        }
        
        # 处理所有信任度题目
        trust_scores = []
        trust_details = {}
        
        for col in trust_columns:
            cleaned_score = clean_trust_score(row[col])
            if not pd.isna(cleaned_score):
                # 反向计分题需要转换（8 - 原分）
                if col in reverse_columns:
                    cleaned_score = 8 - cleaned_score
                
                trust_scores.append(cleaned_score)
                
                # 简化列名作为详细信息的键
                simplified_name = col.replace('73、', '').split('。')[0][:20]
                trust_details[f'trust_{len(trust_details)+1}_{simplified_name}'] = cleaned_score
        
        if len(trust_scores) >= 10:  # 至少需要10个有效回答（12题中的大部分）
            participant_data.update({
                'trust_total_score': sum(trust_scores),
                'trust_avg_score': np.mean(trust_scores),
                'trust_valid_items': len(trust_scores),
                **trust_details
            })
            trust_data.append(participant_data)
    
    trust_df = pd.DataFrame(trust_data)
    print(f"有效参与者数据: {len(trust_df)}个")
    
    return trust_df

def calculate_trust_statistics(trust_df):
    """计算信任度统计分析"""
    print("\n=== 信任度统计分析 ===")
    
    stats_data = []
    
    # 1. 总体统计
    print("\n1. 总体信任度统计:")
    overall_scores = trust_df['trust_avg_score']
    if len(overall_scores) > 0:
        stats_data.append({
            '分析维度': '总体',
            '分组': '所有参与者',
            '样本数': len(overall_scores),
            '平均信任度': round(overall_scores.mean(), 2),
            '标准差': round(overall_scores.std(), 2),
            '中位数': round(overall_scores.median(), 2),
            '最小值': round(overall_scores.min(), 2),
            '最大值': round(overall_scores.max(), 2),
            '信任水平描述': get_trust_description(overall_scores.mean())
        })
        print(f"  总体: n={len(overall_scores)}, M={overall_scores.mean():.2f}, SD={overall_scores.std():.2f}")
    
    # 2. 按科室统计
    print("\n2. 按科室信任度统计:")
    for dept in trust_df['科室'].unique():
        dept_data = trust_df[trust_df['科室'] == dept]
        dept_scores = dept_data['trust_avg_score']
        
        if len(dept_scores) > 0:
            stats_data.append({
                '分析维度': f'科室_{dept}',
                '分组': dept,
                '样本数': len(dept_scores),
                '平均信任度': round(dept_scores.mean(), 2),
                '标准差': round(dept_scores.std(), 2),
                '中位数': round(dept_scores.median(), 2),
                '最小值': round(dept_scores.min(), 2),
                '最大值': round(dept_scores.max(), 2),
                '信任水平描述': get_trust_description(dept_scores.mean())
            })
            print(f"  {dept}: n={len(dept_scores)}, M={dept_scores.mean():.2f}")
    
    # 3. 按年资统计
    print("\n3. 按年资信任度统计:")
    for seniority in trust_df['年资分类'].unique():
        if seniority == '未知':
            continue
            
        seniority_data = trust_df[trust_df['年资分类'] == seniority]
        seniority_scores = seniority_data['trust_avg_score']
        
        if len(seniority_scores) > 0:
            stats_data.append({
                '分析维度': f'年资_{seniority}',
                '分组': seniority,
                '样本数': len(seniority_scores),
                '平均信任度': round(seniority_scores.mean(), 2),
                '标准差': round(seniority_scores.std(), 2),
                '中位数': round(seniority_scores.median(), 2),
                '最小值': round(seniority_scores.min(), 2),
                '最大值': round(seniority_scores.max(), 2),
                '信任水平描述': get_trust_description(seniority_scores.mean())
            })
            print(f"  {seniority}: n={len(seniority_scores)}, M={seniority_scores.mean():.2f}")
    
    # 4. 按AI使用顺序统计
    print("\n4. 按AI使用顺序信任度统计:")
    for ai_first in [True, False]:
        ai_label = 'AI先用' if ai_first else 'AI后用'
        ai_data = trust_df[trust_df['ai_first'] == ai_first]
        ai_scores = ai_data['trust_avg_score']
        
        if len(ai_scores) > 0:
            stats_data.append({
                '分析维度': 'AI使用顺序',
                '分组': ai_label,
                '样本数': len(ai_scores),
                '平均信任度': round(ai_scores.mean(), 2),
                '标准差': round(ai_scores.std(), 2),
                '中位数': round(ai_scores.median(), 2),
                '最小值': round(ai_scores.min(), 2),
                '最大值': round(ai_scores.max(), 2),
                '信任水平描述': get_trust_description(ai_scores.mean())
            })
            print(f"  {ai_label}: n={len(ai_scores)}, M={ai_scores.mean():.2f}")
    
    # 执行AI使用顺序的t检验
    ai_first_scores = trust_df[trust_df['ai_first'] == True]['trust_avg_score']
    ai_second_scores = trust_df[trust_df['ai_first'] == False]['trust_avg_score']
    
    if len(ai_first_scores) > 0 and len(ai_second_scores) > 0:
        t_stat, p_value = stats.ttest_ind(ai_first_scores, ai_second_scores)
        print(f"  AI使用顺序独立t检验: t={t_stat:.4f}, p={p_value:.4f}")
        
        # 添加检验结果到统计数据
        for i, ai_first in enumerate([True, False]):
            ai_label = 'AI先用' if ai_first else 'AI后用'
            matching_stats = [s for s in stats_data if s['分析维度'] == 'AI使用顺序' and s['分组'] == ai_label]
            for stat in matching_stats:
                stat['t_value'] = t_stat
                stat['p_value'] = p_value
                stat['effect_size'] = calculate_cohens_d_independent(ai_first_scores, ai_second_scores)
    
    return pd.DataFrame(stats_data)

def get_trust_description(score):
    """根据分数获取信任水平描述"""
    if pd.isna(score):
        return 'N/A'
    elif score >= 6.5:
        return '非常高信任'
    elif score >= 5.5:
        return '高信任'
    elif score >= 4.5:
        return '中等信任'
    elif score >= 3.5:
        return '较低信任'
    else:
        return '低信任'

def calculate_cohens_d_independent(group1, group2):
    """计算独立样本的Cohen's d效应量"""
    try:
        if len(group1) == 0 or len(group2) == 0:
            return np.nan
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # 合并标准差
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return np.nan
            
        return (mean1 - mean2) / pooled_std
    except:
        return np.nan

def analyze_trust_dimensions(trust_df):
    """分析信任度各维度"""
    print("\n=== 信任度维度分析 ===")
    
    # 获取信任度详细题目列
    trust_detail_cols = [col for col in trust_df.columns if col.startswith('trust_') and col not in ['trust_total_score', 'trust_avg_score', 'trust_valid_items']]
    
    dimension_data = []
    
    for col in trust_detail_cols:
        if col in trust_df.columns:
            # 提取维度名称
            dim_name = col.replace('trust_', '').split('_', 1)[1] if '_' in col[6:] else col
            print(f"\n{dim_name}维度分析:")
            
            scores = trust_df[col].dropna()
            
            if len(scores) > 0:
                dimension_data.append({
                    '信任维度': dim_name,
                    '样本数': len(scores),
                    '平均分': round(scores.mean(), 2),
                    '标准差': round(scores.std(), 2),
                    '中位数': round(scores.median(), 2),
                    '最小值': round(scores.min(), 2),
                    '最大值': round(scores.max(), 2)
                })
                print(f"  M={scores.mean():.2f}, SD={scores.std():.2f}")
    
    return pd.DataFrame(dimension_data)

def main():
    """主函数"""
    print("=== 自动化信任度量表分析 ===")
    
    # 加载数据并创建分组
    df = load_data_and_create_groups()
    
    # 计算信任度得分
    trust_df = calculate_trust_scores(df)
    
    if trust_df.empty:
        print("❌ 无法计算信任度得分，程序退出")
        return
    
    # 计算统计分析
    trust_stats = calculate_trust_statistics(trust_df)
    
    # 分析信任度各维度
    dimension_analysis = analyze_trust_dimensions(trust_df)
    
    # 保存结果文件
    
    # 1. 信任度原始数据
    raw_file = 'results/trust/trust_scale_data.csv'
    trust_df.to_csv(raw_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 信任度原始数据已保存到: {raw_file}")
    
    # 2. 信任度统计分析
    stats_file = 'results/trust/trust_scale_analysis.csv'
    trust_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"✓ 信任度统计分析已保存到: {stats_file}")
    
    # 3. 信任度维度分析
    dimension_file = 'results/trust/trust_scale_dimensions.csv'
    dimension_analysis.to_csv(dimension_file, index=False, encoding='utf-8-sig')
    print(f"✓ 信任度维度分析已保存到: {dimension_file}")
    
    # 显示关键结果概览
    print(f"\n=== 信任度分析结果概览 ===")
    print(f"有效参与者数量: {len(trust_df)}")
    
    # 总体效果
    if len(trust_df) > 0:
        overall_mean = trust_df['trust_avg_score'].mean()
        print(f"\n总体信任度:")
        print(f"  平均信任度: {overall_mean:.2f} ({get_trust_description(overall_mean)})")
        
        # 按科室效果
        if '科室' in trust_df.columns:
            print(f"\n按科室信任度:")
            for dept in trust_df['科室'].unique():
                dept_data = trust_df[trust_df['科室'] == dept]
                dept_mean = dept_data['trust_avg_score'].mean()
                print(f"  {dept}: {dept_mean:.2f} ({get_trust_description(dept_mean)})")
        
        # 按AI使用顺序效果
        ai_first_mean = trust_df[trust_df['ai_first'] == True]['trust_avg_score'].mean()
        ai_second_mean = trust_df[trust_df['ai_first'] == False]['trust_avg_score'].mean()
        
        if not pd.isna(ai_first_mean) and not pd.isna(ai_second_mean):
            print(f"\n按AI使用顺序:")
            print(f"  AI先用: {ai_first_mean:.2f} ({get_trust_description(ai_first_mean)})")
            print(f"  AI后用: {ai_second_mean:.2f} ({get_trust_description(ai_second_mean)})")
            print(f"  差异: {ai_first_mean - ai_second_mean:+.2f}")

if __name__ == "__main__":
    main()
