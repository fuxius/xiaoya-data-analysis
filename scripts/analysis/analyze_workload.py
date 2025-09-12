#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ2分析：认知负荷分析
计算NASA-TLX总分并比较AI辅助组与无辅助组的认知负荷，贯穿不同群体差异分析

作者: AICare研究团队
日期: 2025年9月
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
import re

# 导入rm-ANOVA分析模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from rm_anova_analysis import perform_rm_anova_analysis, print_rm_anova_summary

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

def clean_nasa_score(value):
    """清洗NASA-TLX分数数据"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        if value == '(空)':
            return 0
        # 尝试提取数字
        numbers = re.findall(r'\d+', str(value))
        if numbers:
            return float(numbers[0])
        return np.nan
    return float(value)

def calculate_nasa_tlx_scores(df):
    """计算NASA-TLX得分"""
    print("\n正在计算NASA-TLX得分...")
    
    # NASA-TLX维度名称
    nasa_dimensions = [
        '脑力需求', '体力需求', '时间压力', '任务表现', '努力程度', '挫败感'
    ]
    
    # 找到两组NASA-TLX数据（第一次和第二次测量）
    nasa_cols_condition1 = []  # 第一次测量（条件1）
    nasa_cols_condition2 = []  # 第二次测量（条件2）
    
    for col in df.columns:
        for dim in nasa_dimensions:
            if dim in col:
                # 根据列名中的题号判断是第一组还是第二组
                if any(num in col for num in ['35、', '36、', '37、', '38、', '39、', '40、']):
                    nasa_cols_condition1.append(col)
                elif any(num in col for num in ['66、', '67、', '68、', '69、', '70、', '71、']):
                    nasa_cols_condition2.append(col)
    
    print(f"第一次测量NASA-TLX列: {len(nasa_cols_condition1)}个")
    print(f"第二次测量NASA-TLX列: {len(nasa_cols_condition2)}个")
    
    # 为每个参与者计算NASA-TLX数据
    workload_data = []
    
    for idx, row in df.iterrows():
        participant_data = {
            'ID': row['ID'],
            '科室': row['科室'],
            '年资分类': row['年资分类'],
            'hospital': row['hospital'],
            'ai_first': row['ai_first']
        }
        
        # 处理第一次测量（条件1）
        if len(nasa_cols_condition1) >= 6:
            condition1_scores = []
            condition1_details = {}
            
            for i, col in enumerate(nasa_cols_condition1):
                cleaned_score = clean_nasa_score(row[col])
                if not pd.isna(cleaned_score):
                    condition1_scores.append(cleaned_score)
                    dim_name = nasa_dimensions[i % len(nasa_dimensions)]
                    condition1_details[f'condition1_{dim_name}'] = cleaned_score
            
            if len(condition1_scores) >= 5:  # 至少需要5个有效维度
                participant_data.update({
                    'condition1_total_score': sum(condition1_scores),
                    'condition1_avg_score': np.mean(condition1_scores),
                    'condition1_valid_dimensions': len(condition1_scores),
                    **condition1_details
                })
        
        # 处理第二次测量（条件2）
        if len(nasa_cols_condition2) >= 6:
            condition2_scores = []
            condition2_details = {}
            
            for i, col in enumerate(nasa_cols_condition2):
                cleaned_score = clean_nasa_score(row[col])
                if not pd.isna(cleaned_score):
                    condition2_scores.append(cleaned_score)
                    dim_name = nasa_dimensions[i % len(nasa_dimensions)]
                    condition2_details[f'condition2_{dim_name}'] = cleaned_score
            
            if len(condition2_scores) >= 5:  # 至少需要5个有效维度
                participant_data.update({
                    'condition2_total_score': sum(condition2_scores),
                    'condition2_avg_score': np.mean(condition2_scores),
                    'condition2_valid_dimensions': len(condition2_scores),
                    **condition2_details
                })
        
        # 只保留有有效NASA-TLX数据的参与者
        if 'condition1_total_score' in participant_data or 'condition2_total_score' in participant_data:
            workload_data.append(participant_data)
    
    workload_df = pd.DataFrame(workload_data)
    print(f"有效参与者数据: {len(workload_df)}个")
    
    return workload_df

def assign_conditions_to_scores(workload_df):
    """将条件1和条件2分配给AI辅助和无辅助条件"""
    print("\n正在分配条件...")
    
    # 创建长格式数据，每个参与者有两行（AI辅助和无辅助）
    long_format_data = []
    
    for idx, row in workload_df.iterrows():
        # 根据ai_first确定条件分配
        if row['ai_first']:
            # AI先用：条件1=AI辅助，条件2=无辅助
            if 'condition1_total_score' in row and pd.notna(row['condition1_total_score']):
                long_format_data.append({
                    'ID': row['ID'],
                    '科室': row['科室'],
                    '年资分类': row['年资分类'],
                    'hospital': row['hospital'],
                    'ai_first': row['ai_first'],
                    'condition_type': 'AI辅助组',
                    'nasa_tlx_total': row['condition1_total_score'],
                    'nasa_tlx_avg': row['condition1_avg_score'],
                    'valid_dimensions': row['condition1_valid_dimensions'],
                    '脑力需求': row.get('condition1_脑力需求', np.nan),
                    '体力需求': row.get('condition1_体力需求', np.nan),
                    '时间压力': row.get('condition1_时间压力', np.nan),
                    '任务表现': row.get('condition1_任务表现', np.nan),
                    '努力程度': row.get('condition1_努力程度', np.nan),
                    '挫败感': row.get('condition1_挫败感', np.nan)
                })
            
            if 'condition2_total_score' in row and pd.notna(row['condition2_total_score']):
                long_format_data.append({
                    'ID': row['ID'],
                    '科室': row['科室'],
                    '年资分类': row['年资分类'],
                    'hospital': row['hospital'],
                    'ai_first': row['ai_first'],
                    'condition_type': '无辅助组',
                    'nasa_tlx_total': row['condition2_total_score'],
                    'nasa_tlx_avg': row['condition2_avg_score'],
                    'valid_dimensions': row['condition2_valid_dimensions'],
                    '脑力需求': row.get('condition2_脑力需求', np.nan),
                    '体力需求': row.get('condition2_体力需求', np.nan),
                    '时间压力': row.get('condition2_时间压力', np.nan),
                    '任务表现': row.get('condition2_任务表现', np.nan),
                    '努力程度': row.get('condition2_努力程度', np.nan),
                    '挫败感': row.get('condition2_挫败感', np.nan)
                })
        else:
            # AI后用：条件1=无辅助，条件2=AI辅助
            if 'condition1_total_score' in row and pd.notna(row['condition1_total_score']):
                long_format_data.append({
                    'ID': row['ID'],
                    '科室': row['科室'],
                    '年资分类': row['年资分类'],
                    'hospital': row['hospital'],
                    'ai_first': row['ai_first'],
                    'condition_type': '无辅助组',
                    'nasa_tlx_total': row['condition1_total_score'],
                    'nasa_tlx_avg': row['condition1_avg_score'],
                    'valid_dimensions': row['condition1_valid_dimensions'],
                    '脑力需求': row.get('condition1_脑力需求', np.nan),
                    '体力需求': row.get('condition1_体力需求', np.nan),
                    '时间压力': row.get('condition1_时间压力', np.nan),
                    '任务表现': row.get('condition1_任务表现', np.nan),
                    '努力程度': row.get('condition1_努力程度', np.nan),
                    '挫败感': row.get('condition1_挫败感', np.nan)
                })
            
            if 'condition2_total_score' in row and pd.notna(row['condition2_total_score']):
                long_format_data.append({
                    'ID': row['ID'],
                    '科室': row['科室'],
                    '年资分类': row['年资分类'],
                    'hospital': row['hospital'],
                    'ai_first': row['ai_first'],
                    'condition_type': 'AI辅助组',
                    'nasa_tlx_total': row['condition2_total_score'],
                    'nasa_tlx_avg': row['condition2_avg_score'],
                    'valid_dimensions': row['condition2_valid_dimensions'],
                    '脑力需求': row.get('condition2_脑力需求', np.nan),
                    '体力需求': row.get('condition2_体力需求', np.nan),
                    '时间压力': row.get('condition2_时间压力', np.nan),
                    '任务表现': row.get('condition2_任务表现', np.nan),
                    '努力程度': row.get('condition2_努力程度', np.nan),
                    '挫败感': row.get('condition2_挫败感', np.nan)
                })
    
    workload_long_df = pd.DataFrame(long_format_data)
    print(f"长格式数据记录数: {len(workload_long_df)}个")
    
    return workload_long_df

def calculate_workload_statistics(workload_df):
    """计算认知负荷统计分析"""
    print("\n=== 认知负荷统计分析 ===")
    
    stats_data = []
    
    # 1. 总体统计
    print("\n1. 总体认知负荷统计:")
    for condition in ['AI辅助组', '无辅助组']:
        condition_data = workload_df[workload_df['condition_type'] == condition]
        if len(condition_data) > 0:
            nasa_scores = condition_data['nasa_tlx_total']
            stats_data.append({
                '分析维度': '总体',
                '分组': condition,
                '样本数': len(nasa_scores),
                '平均NASA-TLX总分': round(nasa_scores.mean(), 2),
                '标准差': round(nasa_scores.std(), 2),
                '中位数': round(nasa_scores.median(), 2),
                '最小值': round(nasa_scores.min(), 2),
                '最大值': round(nasa_scores.max(), 2),
                '负荷水平描述': get_workload_description(nasa_scores.mean())
            })
            print(f"  {condition}: n={len(nasa_scores)}, M={nasa_scores.mean():.2f}, SD={nasa_scores.std():.2f}")
    
    # 执行总体rm-ANOVA
    print(f"\n执行总体rm-ANOVA分析:")
    rm_results = perform_rm_anova_analysis(
        workload_df,
        participant_col='ID',
        condition_col='condition_type',
        dv_col='nasa_tlx_total'
    )
    
    if rm_results:
        print_rm_anova_summary(rm_results, "总体认知负荷分析")
        
        # 提取统计量
        main_effect = rm_results.get('main_effect', pd.DataFrame())
        effect_size = rm_results.get('effect_size_pes', pd.DataFrame())
        
        f_value = main_effect['F'].iloc[0] if len(main_effect) > 0 and 'F' in main_effect.columns else np.nan
        p_value = main_effect['Pr(>F)'].iloc[0] if len(main_effect) > 0 and 'Pr(>F)' in main_effect.columns else np.nan
        pes_value = effect_size['pes'].iloc[0] if len(effect_size) > 0 and 'pes' in effect_size.columns else np.nan
        
        # 添加检验结果到统计数据
        for condition in ['AI辅助组', '无辅助组']:
            matching_stats = [s for s in stats_data if s['分析维度'] == '总体' and s['分组'] == condition]
            for stat in matching_stats:
                stat['f_value'] = f_value
                stat['p_value'] = p_value
                stat['partial_eta_squared'] = pes_value
                stat['analysis_type'] = rm_results.get('analysis_type', 'RM_ANOVA')
    
    # 2. 按科室统计
    print("\n2. 按科室认知负荷统计:")
    for dept in workload_df['科室'].unique():
        dept_data = workload_df[workload_df['科室'] == dept]
        print(f"\n  {dept}:")
        
        for condition in ['AI辅助组', '无辅助组']:
            condition_dept_data = dept_data[dept_data['condition_type'] == condition]
            if len(condition_dept_data) > 0:
                nasa_scores = condition_dept_data['nasa_tlx_total']
                stats_data.append({
                    '分析维度': f'科室_{dept}',
                    '分组': condition,
                    '样本数': len(nasa_scores),
                    '平均NASA-TLX总分': round(nasa_scores.mean(), 2),
                    '标准差': round(nasa_scores.std(), 2),
                    '中位数': round(nasa_scores.median(), 2),
                    '最小值': round(nasa_scores.min(), 2),
                    '最大值': round(nasa_scores.max(), 2),
                    '负荷水平描述': get_workload_description(nasa_scores.mean())
                })
                print(f"    {condition}: n={len(nasa_scores)}, M={nasa_scores.mean():.2f}")
        
        # 科室内rm-ANOVA
        if len(dept_data) >= 4:  # 至少需要2个参与者×2个条件
            print(f"    执行{dept}科室rm-ANOVA:")
            dept_rm_results = perform_rm_anova_analysis(
                dept_data,
                participant_col='ID',
                condition_col='condition_type',
                dv_col='nasa_tlx_total'
            )
            
            if dept_rm_results:
                print_rm_anova_summary(dept_rm_results, f"{dept}科室认知负荷分析")
                
                # 提取统计量
                main_effect = dept_rm_results.get('main_effect', pd.DataFrame())
                effect_size = dept_rm_results.get('effect_size_pes', pd.DataFrame())
                
                f_value = main_effect['F'].iloc[0] if len(main_effect) > 0 and 'F' in main_effect.columns else np.nan
                p_value = main_effect['Pr(>F)'].iloc[0] if len(main_effect) > 0 and 'Pr(>F)' in main_effect.columns else np.nan
                pes_value = effect_size['pes'].iloc[0] if len(effect_size) > 0 and 'pes' in effect_size.columns else np.nan
                
                # 添加检验结果
                dept_stats = [s for s in stats_data if s['分析维度'] == f'科室_{dept}']
                for stat in dept_stats:
                    stat['f_value'] = f_value
                    stat['p_value'] = p_value
                    stat['partial_eta_squared'] = pes_value
                    stat['analysis_type'] = dept_rm_results.get('analysis_type', 'RM_ANOVA')
            else:
                print(f"    {dept}科室rm-ANOVA分析失败")
        else:
            print(f"    {dept}科室: 数据不足，跳过rm-ANOVA分析")
    
    # 3. 按年资统计
    print("\n3. 按年资认知负荷统计:")
    for seniority in workload_df['年资分类'].unique():
        if seniority == '未知':
            continue
            
        seniority_data = workload_df[workload_df['年资分类'] == seniority]
        print(f"\n  {seniority}:")
        
        for condition in ['AI辅助组', '无辅助组']:
            condition_seniority_data = seniority_data[seniority_data['condition_type'] == condition]
            if len(condition_seniority_data) > 0:
                nasa_scores = condition_seniority_data['nasa_tlx_total']
                stats_data.append({
                    '分析维度': f'年资_{seniority}',
                    '分组': condition,
                    '样本数': len(nasa_scores),
                    '平均NASA-TLX总分': round(nasa_scores.mean(), 2),
                    '标准差': round(nasa_scores.std(), 2),
                    '中位数': round(nasa_scores.median(), 2),
                    '最小值': round(nasa_scores.min(), 2),
                    '最大值': round(nasa_scores.max(), 2),
                    '负荷水平描述': get_workload_description(nasa_scores.mean())
                })
                print(f"    {condition}: n={len(nasa_scores)}, M={nasa_scores.mean():.2f}")
        
        # 年资内rm-ANOVA
        if len(seniority_data) >= 4:  # 至少需要2个参与者×2个条件
            print(f"    执行{seniority}年资rm-ANOVA:")
            seniority_rm_results = perform_rm_anova_analysis(
                seniority_data,
                participant_col='ID',
                condition_col='condition_type',
                dv_col='nasa_tlx_total'
            )
            
            if seniority_rm_results:
                print_rm_anova_summary(seniority_rm_results, f"{seniority}年资认知负荷分析")
                
                # 提取统计量
                main_effect = seniority_rm_results.get('main_effect', pd.DataFrame())
                effect_size = seniority_rm_results.get('effect_size_pes', pd.DataFrame())
                
                f_value = main_effect['F'].iloc[0] if len(main_effect) > 0 and 'F' in main_effect.columns else np.nan
                p_value = main_effect['Pr(>F)'].iloc[0] if len(main_effect) > 0 and 'Pr(>F)' in main_effect.columns else np.nan
                pes_value = effect_size['pes'].iloc[0] if len(effect_size) > 0 and 'pes' in effect_size.columns else np.nan
                
                # 添加检验结果
                seniority_stats = [s for s in stats_data if s['分析维度'] == f'年资_{seniority}']
                for stat in seniority_stats:
                    stat['f_value'] = f_value
                    stat['p_value'] = p_value
                    stat['partial_eta_squared'] = pes_value
                    stat['analysis_type'] = seniority_rm_results.get('analysis_type', 'RM_ANOVA')
            else:
                print(f"    {seniority}年资rm-ANOVA分析失败")
        else:
            print(f"    {seniority}年资: 数据不足，跳过rm-ANOVA分析")
    
    return pd.DataFrame(stats_data)

def get_paired_data(workload_df):
    """获取配对数据"""
    paired_data = []
    
    # 按ID分组，找到同时有AI辅助和无辅助数据的参与者
    for participant_id in workload_df['ID'].unique():
        participant_data = workload_df[workload_df['ID'] == participant_id]
        
        ai_data = participant_data[participant_data['condition_type'] == 'AI辅助组']
        no_ai_data = participant_data[participant_data['condition_type'] == '无辅助组']
        
        if len(ai_data) > 0 and len(no_ai_data) > 0:
            paired_data.append({
                'ID': participant_id,
                'ai_score': ai_data.iloc[0]['nasa_tlx_total'],
                'no_ai_score': no_ai_data.iloc[0]['nasa_tlx_total']
            })
    
    return paired_data

def get_workload_description(score):
    """根据分数获取认知负荷水平描述"""
    if pd.isna(score):
        return 'N/A'
    elif score >= 300:
        return '极高负荷'
    elif score >= 250:
        return '高负荷'
    elif score >= 200:
        return '中等负荷'
    elif score >= 150:
        return '较低负荷'
    else:
        return '低负荷'

def calculate_cohens_d_paired(group1, group2):
    """计算配对样本的Cohen's d效应量"""
    try:
        differences = np.array(group1) - np.array(group2)
        if len(differences) == 0:
            return np.nan
        
        diff_std = differences.std()
        if diff_std == 0:
            return np.nan
            
        return differences.mean() / diff_std
    except:
        return np.nan

def analyze_nasa_dimensions(workload_df):
    """分析NASA-TLX各维度"""
    print("\n=== NASA-TLX维度分析 ===")
    
    dimensions = ['脑力需求', '体力需求', '时间压力', '任务表现', '努力程度', '挫败感']
    dimension_data = []
    
    for dim in dimensions:
        if dim in workload_df.columns:
            print(f"\n{dim}维度分析:")
            
            for condition in ['AI辅助组', '无辅助组']:
                condition_data = workload_df[workload_df['condition_type'] == condition]
                dim_scores = condition_data[dim].dropna()
                
                if len(dim_scores) > 0:
                    dimension_data.append({
                        'NASA-TLX维度': dim,
                        '条件': condition,
                        '样本数': len(dim_scores),
                        '平均分': round(dim_scores.mean(), 2),
                        '标准差': round(dim_scores.std(), 2),
                        '中位数': round(dim_scores.median(), 2)
                    })
                    print(f"  {condition}: M={dim_scores.mean():.2f}, SD={dim_scores.std():.2f}")
            
            # 维度配对t检验
            paired_data = []
            for participant_id in workload_df['ID'].unique():
                participant_data = workload_df[workload_df['ID'] == participant_id]
                ai_data = participant_data[participant_data['condition_type'] == 'AI辅助组']
                no_ai_data = participant_data[participant_data['condition_type'] == '无辅助组']
                
                if len(ai_data) > 0 and len(no_ai_data) > 0:
                    ai_score = ai_data.iloc[0][dim]
                    no_ai_score = no_ai_data.iloc[0][dim]
                    
                    if pd.notna(ai_score) and pd.notna(no_ai_score):
                        paired_data.append({
                            'ai_score': ai_score,
                            'no_ai_score': no_ai_score
                        })
            
            if len(paired_data) > 1:
                # 准备长格式数据用于rm-ANOVA
                long_format_data = []
                for i, pair in enumerate(paired_data):
                    long_format_data.append({
                        'participant_id': f'P{i+1}',
                        'condition': 'AI辅助',
                        'workload_score': pair['ai_score']
                    })
                    long_format_data.append({
                        'participant_id': f'P{i+1}',
                        'condition': '无辅助',
                        'workload_score': pair['no_ai_score']
                    })
                
                workload_long_df = pd.DataFrame(long_format_data)
                
                try:
                    # 执行rm-ANOVA分析
                    rm_results = perform_rm_anova_analysis(
                        workload_long_df,
                        participant_col='participant_id',
                        condition_col='condition',
                        dv_col='workload_score'
                    )
                    
                    if rm_results:
                        print_rm_anova_summary(rm_results, f"{dim}维度工作负荷分析")
                        
                        # 提取统计量
                        main_effect = rm_results.get('main_effect', pd.DataFrame())
                        effect_size = rm_results.get('effect_size_pes', pd.DataFrame())
                        
                        f_value = main_effect['F'].iloc[0] if len(main_effect) > 0 and 'F' in main_effect.columns else np.nan
                        p_value = main_effect['Pr(>F)'].iloc[0] if len(main_effect) > 0 and 'Pr(>F)' in main_effect.columns else np.nan
                        pes_value = effect_size['pes'].iloc[0] if len(effect_size) > 0 and 'pes' in effect_size.columns else np.nan
                        
                        # 更新维度数据
                        for data in dimension_data:
                            if data['NASA-TLX维度'] == dim:
                                data['f_value'] = f_value
                                data['p_value'] = p_value
                                data['partial_eta_squared'] = pes_value
                                data['analysis_type'] = 'RM_ANOVA'
                    else:
                        print(f"  {dim}rm-ANOVA分析失败")
                except Exception as e:
                    print(f"  {dim}rm-ANOVA分析失败: {e}")
    
    return pd.DataFrame(dimension_data)

def main():
    """主函数"""
    print("=== RQ2: 认知负荷分析 ===")
    
    # 加载数据并创建分组
    df = load_data_and_create_groups()
    
    # 计算NASA-TLX得分
    workload_df = calculate_nasa_tlx_scores(df)
    
    # 分配条件
    workload_long_df = assign_conditions_to_scores(workload_df)
    
    # 计算统计分析
    workload_stats = calculate_workload_statistics(workload_long_df)
    
    # 分析NASA-TLX各维度
    dimension_analysis = analyze_nasa_dimensions(workload_long_df)
    
    # 保存结果文件
    
    # 1. 原始认知负荷数据（宽格式）
    raw_file = 'results/workload/workload_raw_data.csv'
    workload_df.to_csv(raw_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 认知负荷原始数据已保存到: {raw_file}")
    
    # 2. 长格式认知负荷数据
    long_file = 'results/workload/workload_long_data.csv'
    workload_long_df.to_csv(long_file, index=False, encoding='utf-8-sig')
    print(f"✓ 认知负荷长格式数据已保存到: {long_file}")
    
    # 3. 认知负荷统计分析
    stats_file = 'results/workload/workload_analysis.csv'
    workload_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"✓ 认知负荷统计分析已保存到: {stats_file}")
    
    # 4. NASA-TLX维度分析
    dimension_file = 'results/workload/nasa_tlx_dimensions.csv'
    dimension_analysis.to_csv(dimension_file, index=False, encoding='utf-8-sig')
    print(f"✓ NASA-TLX维度分析已保存到: {dimension_file}")
    
    # 显示关键结果概览
    print(f"\n=== 认知负荷分析结果概览 ===")
    print(f"有效参与者数量: {len(workload_df)}")
    print(f"配对数据记录数: {len(workload_long_df)}")
    
    # 总体效果
    ai_group = workload_long_df[workload_long_df['condition_type'] == 'AI辅助组']
    no_ai_group = workload_long_df[workload_long_df['condition_type'] == '无辅助组']
    
    if len(ai_group) > 0 and len(no_ai_group) > 0:
        ai_mean = ai_group['nasa_tlx_total'].mean()
        no_ai_mean = no_ai_group['nasa_tlx_total'].mean()
        workload_diff = ai_mean - no_ai_mean
        
        print(f"\n总体效果:")
        print(f"  AI辅助组平均负荷: {ai_mean:.2f} ({get_workload_description(ai_mean)})")
        print(f"  无辅助组平均负荷: {no_ai_mean:.2f} ({get_workload_description(no_ai_mean)})")
        print(f"  负荷差异: {workload_diff:+.2f}")
        
        # 效果解释
        if abs(workload_diff) < 10:
            effect_desc = "无明显差异"
        elif workload_diff > 0:
            effect_desc = "AI辅助增加认知负荷"
        else:
            effect_desc = "AI辅助减少认知负荷"
        
        print(f"  效果评价: {effect_desc}")
    
    # 按科室效果
    if '科室' in workload_long_df.columns:
        print(f"\n按科室效果:")
        for dept in workload_long_df['科室'].unique():
            dept_data = workload_long_df[workload_long_df['科室'] == dept]
            dept_ai = dept_data[dept_data['condition_type'] == 'AI辅助组']['nasa_tlx_total']
            dept_no_ai = dept_data[dept_data['condition_type'] == '无辅助组']['nasa_tlx_total']
            
            if len(dept_ai) > 0 and len(dept_no_ai) > 0:
                diff = dept_ai.mean() - dept_no_ai.mean()
                print(f"  {dept}: AI vs 无AI = {dept_ai.mean():.1f} vs {dept_no_ai.mean():.1f} (差异: {diff:+.1f})")

if __name__ == "__main__":
    main()
