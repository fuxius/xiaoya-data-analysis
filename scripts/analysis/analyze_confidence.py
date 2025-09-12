#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ2分析：诊断信心分析
编码信心水平并比较AI辅助组与无辅助组的诊断信心，贯穿不同群体差异分析

作者: AICare研究团队
日期: 2025年9月
"""

import pandas as pd
import numpy as np
import json
from scipy import stats

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

def encode_confidence_levels(df):
    """编码信心水平 - 被试内设计"""
    print("\n正在编码信心水平...")
    
    # 信心水平映射
    confidence_mapping = {
        '没有信心': 1,
        '略有信心': 2,
        '信心一般': 3,
        '比较有信心': 4,
        '非常有信心': 5,
        '(空)': np.nan
    }
    
    # 找到所有信心相关的列
    confidence_cols = [col for col in df.columns if '信心' in col and '多大' in col]
    confidence_cols = sorted(confidence_cols)  # 确保顺序正确
    print(f"找到信心评估列: {len(confidence_cols)}个")
    print(f"信心评估列: {confidence_cols}")
    
    # 为每个参与者提取信心数据（宽格式）
    confidence_data = []
    
    for idx, row in df.iterrows():
        # 根据条件分配规则提取信心数据
        ai_confidence_scores = []
        no_ai_confidence_scores = []
        
        # 根据ai_first确定哪些案例对应哪种条件
        if row['ai_first']:
            # 先用AI: 前5个案例是AI辅助，后5个案例是无辅助
            ai_indices = list(range(0, min(5, len(confidence_cols))))
            no_ai_indices = list(range(5, len(confidence_cols)))
        else:
            # 后用AI: 前5个案例是无辅助，后5个案例是AI辅助
            no_ai_indices = list(range(0, min(5, len(confidence_cols))))
            ai_indices = list(range(5, len(confidence_cols)))
        
        # 提取AI辅助条件的信心分数
        for i in ai_indices:
            if i < len(confidence_cols):
                col = confidence_cols[i]
                if pd.notna(row[col]) and row[col] != '(空)':
                    score = confidence_mapping.get(row[col], np.nan)
                    if not pd.isna(score):
                        ai_confidence_scores.append(score)
        
        # 提取无辅助条件的信心分数
        for i in no_ai_indices:
            if i < len(confidence_cols):
                col = confidence_cols[i]
                if pd.notna(row[col]) and row[col] != '(空)':
                    score = confidence_mapping.get(row[col], np.nan)
                    if not pd.isna(score):
                        no_ai_confidence_scores.append(score)
        
        # 只有当两种条件都有数据时才包含此参与者
        if len(ai_confidence_scores) > 0 and len(no_ai_confidence_scores) > 0:
            confidence_data.append({
                'ID': row['ID'],
                '科室': row['科室'],
                '年资分类': row['年资分类'],
                'hospital': row['hospital'],
                'ai_first': row['ai_first'],
                'ai_case_count': len(ai_confidence_scores),
                'no_ai_case_count': len(no_ai_confidence_scores),
                'ai_avg_confidence': np.mean(ai_confidence_scores),
                'no_ai_avg_confidence': np.mean(no_ai_confidence_scores),
                'ai_confidence_std': np.std(ai_confidence_scores) if len(ai_confidence_scores) > 1 else 0,
                'no_ai_confidence_std': np.std(no_ai_confidence_scores) if len(no_ai_confidence_scores) > 1 else 0,
                'confidence_difference': np.mean(ai_confidence_scores) - np.mean(no_ai_confidence_scores)
            })
    
    confidence_df = pd.DataFrame(confidence_data)
    print(f"有效参与者数据: {len(confidence_df)}个（具有配对数据）")
    
    return confidence_df

def create_long_format_confidence_data(confidence_df):
    """创建长格式信心数据用于分组分析"""
    print("\n正在创建长格式信心数据...")
    
    long_format_data = []
    
    for idx, row in confidence_df.iterrows():
        # AI辅助条件
        ai_data = {
            'ID': row['ID'],
            '科室': row['科室'],
            '年资分类': row['年资分类'],
            'hospital': row['hospital'],
            'ai_first': row['ai_first'],
            'condition_type': 'AI辅助组',
            'avg_confidence': row['ai_avg_confidence'],
            'case_count': row['ai_case_count']
        }
        long_format_data.append(ai_data)
        
        # 无辅助条件
        no_ai_data = {
            'ID': row['ID'],
            '科室': row['科室'],
            '年资分类': row['年资分类'],
            'hospital': row['hospital'],
            'ai_first': row['ai_first'],
            'condition_type': '无辅助组',
            'avg_confidence': row['no_ai_avg_confidence'],
            'case_count': row['no_ai_case_count']
        }
        long_format_data.append(no_ai_data)
    
    confidence_long_df = pd.DataFrame(long_format_data)
    print(f"长格式数据记录数: {len(confidence_long_df)}个")
    
    return confidence_long_df

def calculate_confidence_statistics(confidence_df):
    """计算诊断信心统计分析 - 使用rm-ANOVA替代配对t检验"""
    print("\n=== 诊断信心统计分析 (rm-ANOVA) ===")
    
    stats_data = []
    
    # 准备长格式数据用于rm-ANOVA
    long_format_data = []
    for idx, row in confidence_df.iterrows():
        # AI辅助条件
        long_format_data.append({
            'participant_id': row['ID'],
            'condition': 'AI辅助',
            'confidence_score': row['ai_avg_confidence'],
            'seniority': row['年资分类'],
            'department': row['科室'],
            'hospital': row['hospital']
        })
        
        # 无辅助条件
        long_format_data.append({
            'participant_id': row['ID'],
            'condition': '无辅助',
            'confidence_score': row['no_ai_avg_confidence'],
            'seniority': row['年资分类'],
            'department': row['科室'],
            'hospital': row['hospital']
        })
    
    confidence_long_df = pd.DataFrame(long_format_data)
    
    # 1. 总体统计
    print("\n1. 总体信心统计:")
    
    # 描述性统计
    ai_data = confidence_long_df[confidence_long_df['condition'] == 'AI辅助']
    no_ai_data = confidence_long_df[confidence_long_df['condition'] == '无辅助']
    
    for condition, data in [('AI辅助组', ai_data), ('无辅助组', no_ai_data)]:
        confidence_scores = data['confidence_score']
        stats_data.append({
            '分析维度': '总体',
            '分组': condition,
            '样本数': len(confidence_scores),
            '平均信心得分': round(confidence_scores.mean(), 3),
            '标准差': round(confidence_scores.std(), 3),
            '中位数': round(confidence_scores.median(), 3),
            '最小值': round(confidence_scores.min(), 3),
            '最大值': round(confidence_scores.max(), 3),
            '信心水平描述': get_confidence_description(confidence_scores.mean())
        })
        print(f"  {condition}: n={len(confidence_scores)}, M={confidence_scores.mean():.3f}, SD={confidence_scores.std():.3f}")
    
    # 执行rm-ANOVA
    print(f"\n执行总体rm-ANOVA分析:")
    rm_results = perform_rm_anova_analysis(
        confidence_long_df,
        participant_col='participant_id',
        condition_col='condition',
        dv_col='confidence_score'
    )
    
    if rm_results:
        print_rm_anova_summary(rm_results, "总体信心分析")
        
        # 提取统计量
        main_effect = rm_results.get('main_effect', pd.DataFrame())
        effect_size = rm_results.get('effect_size_pes', pd.DataFrame())
        
        f_value = main_effect['F'].iloc[0] if len(main_effect) > 0 and 'F' in main_effect.columns else np.nan
        p_value = main_effect['Pr(>F)'].iloc[0] if len(main_effect) > 0 and 'Pr(>F)' in main_effect.columns else np.nan
        pes_value = effect_size['pes'].iloc[0] if len(effect_size) > 0 and 'pes' in effect_size.columns else np.nan
        
        # 添加检验结果到统计数据
        for stat in stats_data:
            if stat['分析维度'] == '总体':
                stat['f_value'] = f_value
                stat['p_value'] = p_value
                stat['partial_eta_squared'] = pes_value
                stat['analysis_type'] = rm_results.get('analysis_type', 'RM_ANOVA')
    
    # 2. 按科室统计
    print("\n2. 按科室信心统计:")
    for dept in confidence_df['科室'].unique():
        dept_long_data = confidence_long_df[confidence_long_df['department'] == dept]
        
        if len(dept_long_data) < 4:  # 至少需要2个参与者×2个条件
            print(f"  {dept}: 数据不足，跳过分析")
            continue
            
        print(f"\n  {dept}:")
        
        # 描述性统计
        for condition in ['AI辅助', '无辅助']:
            condition_data = dept_long_data[dept_long_data['condition'] == condition]
            confidence_scores = condition_data['confidence_score']
            
            condition_label = f'{condition}组'
            stats_data.append({
                '分析维度': f'科室_{dept}',
                '分组': condition_label,
                '样本数': len(confidence_scores),
                '平均信心得分': round(confidence_scores.mean(), 3),
                '标准差': round(confidence_scores.std(), 3),
                '中位数': round(confidence_scores.median(), 3),
                '最小值': round(confidence_scores.min(), 3),
                '最大值': round(confidence_scores.max(), 3),
                '信心水平描述': get_confidence_description(confidence_scores.mean())
            })
            print(f"    {condition_label}: n={len(confidence_scores)}, M={confidence_scores.mean():.3f}")
        
        # 科室内rm-ANOVA
        print(f"    执行{dept}科室rm-ANOVA:")
        dept_rm_results = perform_rm_anova_analysis(
            dept_long_data,
            participant_col='participant_id',
            condition_col='condition',
            dv_col='confidence_score'
        )
        
        if dept_rm_results:
            print_rm_anova_summary(dept_rm_results, f"{dept}科室信心分析")
            
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
    
    # 3. 按年资统计
    print("\n3. 按年资信心统计:")
    for seniority in confidence_df['年资分类'].unique():
        if seniority == '未知':
            continue
            
        seniority_long_data = confidence_long_df[confidence_long_df['seniority'] == seniority]
        
        if len(seniority_long_data) < 4:  # 至少需要2个参与者×2个条件
            print(f"  {seniority}: 数据不足，跳过分析")
            continue
            
        print(f"\n  {seniority}:")
        
        # 描述性统计
        for condition in ['AI辅助', '无辅助']:
            condition_data = seniority_long_data[seniority_long_data['condition'] == condition]
            confidence_scores = condition_data['confidence_score']
            
            condition_label = f'{condition}组'
            stats_data.append({
                '分析维度': f'年资_{seniority}',
                '分组': condition_label,
                '样本数': len(confidence_scores),
                '平均信心得分': round(confidence_scores.mean(), 3),
                '标准差': round(confidence_scores.std(), 3),
                '中位数': round(confidence_scores.median(), 3),
                '最小值': round(confidence_scores.min(), 3),
                '最大值': round(confidence_scores.max(), 3),
                '信心水平描述': get_confidence_description(confidence_scores.mean())
            })
            print(f"    {condition_label}: n={len(confidence_scores)}, M={confidence_scores.mean():.3f}")
        
        # 年资内rm-ANOVA
        print(f"    执行{seniority}年资rm-ANOVA:")
        seniority_rm_results = perform_rm_anova_analysis(
            seniority_long_data,
            participant_col='participant_id',
            condition_col='condition',
            dv_col='confidence_score'
        )
        
        if seniority_rm_results:
            print_rm_anova_summary(seniority_rm_results, f"{seniority}年资信心分析")
            
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
    
    # 4. 按医院统计
    print("\n4. 按医院信心统计:")
    for hospital in confidence_df['hospital'].unique():
        hospital_long_data = confidence_long_df[confidence_long_df['hospital'] == hospital]
        
        if len(hospital_long_data) < 4:  # 至少需要2个参与者×2个条件
            print(f"  {hospital}医院: 数据不足，跳过分析")
            continue
            
        print(f"\n  {hospital}医院:")
        
        # 描述性统计
        for condition in ['AI辅助', '无辅助']:
            condition_data = hospital_long_data[hospital_long_data['condition'] == condition]
            confidence_scores = condition_data['confidence_score']
            
            condition_label = f'{condition}组'
            stats_data.append({
                '分析维度': f'医院_{hospital}',
                '分组': condition_label,
                '样本数': len(confidence_scores),
                '平均信心得分': round(confidence_scores.mean(), 3),
                '标准差': round(confidence_scores.std(), 3),
                '中位数': round(confidence_scores.median(), 3),
                '最小值': round(confidence_scores.min(), 3),
                '最大值': round(confidence_scores.max(), 3),
                '信心水平描述': get_confidence_description(confidence_scores.mean())
            })
            print(f"    {condition_label}: n={len(confidence_scores)}, M={confidence_scores.mean():.3f}")
        
        # 医院内rm-ANOVA
        print(f"    执行{hospital}医院rm-ANOVA:")
        hospital_rm_results = perform_rm_anova_analysis(
            hospital_long_data,
            participant_col='participant_id',
            condition_col='condition',
            dv_col='confidence_score'
        )
        
        if hospital_rm_results:
            print_rm_anova_summary(hospital_rm_results, f"{hospital}医院信心分析")
            
            # 提取统计量
            main_effect = hospital_rm_results.get('main_effect', pd.DataFrame())
            effect_size = hospital_rm_results.get('effect_size_pes', pd.DataFrame())
            
            f_value = main_effect['F'].iloc[0] if len(main_effect) > 0 and 'F' in main_effect.columns else np.nan
            p_value = main_effect['Pr(>F)'].iloc[0] if len(main_effect) > 0 and 'Pr(>F)' in main_effect.columns else np.nan
            pes_value = effect_size['pes'].iloc[0] if len(effect_size) > 0 and 'pes' in effect_size.columns else np.nan
            
            # 添加检验结果
            hospital_stats = [s for s in stats_data if s['分析维度'] == f'医院_{hospital}']
            for stat in hospital_stats:
                stat['f_value'] = f_value
                stat['p_value'] = p_value
                stat['partial_eta_squared'] = pes_value
                stat['analysis_type'] = hospital_rm_results.get('analysis_type', 'RM_ANOVA')
        else:
            print(f"    {hospital}医院rm-ANOVA分析失败")
    
    return pd.DataFrame(stats_data)

def get_confidence_description(score):
    """根据分数获取信心水平描述"""
    if pd.isna(score):
        return 'N/A'
    elif score >= 4.5:
        return '非常有信心'
    elif score >= 3.5:
        return '比较有信心'
    elif score >= 2.5:
        return '信心一般'
    elif score >= 1.5:
        return '略有信心'
    else:
        return '没有信心'

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

def analyze_confidence_patterns(confidence_df):
    """分析信心模式"""
    print("\n=== 信心模式分析 ===")
    
    pattern_data = []
    
    # 1. 信心分布分析
    print("\n1. 信心分布分析:")
    for condition in ['AI辅助组', '无辅助组']:
        condition_data = confidence_df[confidence_df['condition_type'] == condition]
        if len(condition_data) > 0:
            confidence_scores = condition_data['avg_confidence']
            
            # 按信心水平分组统计
            confidence_counts = {}
            for score in confidence_scores:
                level = get_confidence_description(score)
                confidence_counts[level] = confidence_counts.get(level, 0) + 1
            
            print(f"\n  {condition}信心分布:")
            for level, count in confidence_counts.items():
                percentage = count / len(condition_data) * 100
                print(f"    {level}: {count}人 ({percentage:.1f}%)")
                
                pattern_data.append({
                    '分析类型': '信心分布',
                    '条件': condition,
                    '信心水平': level,
                    '人数': count,
                    '百分比': round(percentage, 1)
                })
    
    # 2. 信心变异性分析（基于原始宽格式数据）
    print("\n2. 信心变异性分析:")
    # 注意：这里需要使用原始的宽格式数据来计算变异性
    # 我们需要重新从主函数传入正确的数据，这里先简化处理
    print("  变异性分析需要原始配对数据，暂时跳过此部分分析")
    
    return pd.DataFrame(pattern_data)

def main():
    """主函数"""
    print("=== RQ2: 诊断信心分析 (被试内设计) ===")
    
    # 加载数据并创建分组
    df = load_data_and_create_groups()
    
    # 编码信心水平（配对数据）
    confidence_df = encode_confidence_levels(df)
    
    # 创建长格式数据用于分组分析
    confidence_long_df = create_long_format_confidence_data(confidence_df)
    
    # 计算统计分析
    confidence_stats = calculate_confidence_statistics(confidence_df)
    
    # 分析信心模式
    confidence_patterns = analyze_confidence_patterns(confidence_long_df)
    
    # 保存结果文件
    
    # 1. 原始信心数据（宽格式，配对数据）
    raw_file = 'results/confidence/confidence_raw_data.csv'
    confidence_df.to_csv(raw_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 信心原始数据（配对）已保存到: {raw_file}")
    
    # 2. 长格式信心数据
    long_file = 'results/confidence/confidence_long_data.csv'
    confidence_long_df.to_csv(long_file, index=False, encoding='utf-8-sig')
    print(f"✓ 信心长格式数据已保存到: {long_file}")
    
    # 3. 信心统计分析
    stats_file = 'results/confidence/confidence_analysis.csv'
    confidence_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"✓ 信心统计分析已保存到: {stats_file}")
    
    # 4. 信心模式分析
    patterns_file = 'results/confidence/confidence_patterns.csv'
    confidence_patterns.to_csv(patterns_file, index=False, encoding='utf-8-sig')
    print(f"✓ 信心模式分析已保存到: {patterns_file}")
    
    # 显示关键结果概览
    print(f"\n=== 诊断信心分析结果概览 ===")
    print(f"有效配对参与者数量: {len(confidence_df)}")
    
    # 总体效果（配对数据）
    if len(confidence_df) > 0:
        ai_mean = confidence_df['ai_avg_confidence'].mean()
        no_ai_mean = confidence_df['no_ai_avg_confidence'].mean()
        confidence_diff = ai_mean - no_ai_mean
        
        print(f"\n总体效果（配对分析）:")
        print(f"  AI辅助组平均信心: {ai_mean:.3f} ({get_confidence_description(ai_mean)})")
        print(f"  无辅助组平均信心: {no_ai_mean:.3f} ({get_confidence_description(no_ai_mean)})")
        print(f"  信心差异: {confidence_diff:+.3f}")
        
        # 效果解释
        if abs(confidence_diff) < 0.1:
            effect_desc = "无明显差异"
        elif confidence_diff > 0:
            effect_desc = "AI辅助提升信心"
        else:
            effect_desc = "AI辅助降低信心"
        
        print(f"  效果评价: {effect_desc}")
        
        # rm-ANOVA结果已在统计分析中显示
        print(f"  rm-ANOVA结果详见上方统计分析")
    
    # 按科室效果（配对分析）
    if '科室' in confidence_df.columns:
        print(f"\n按科室效果（配对分析）:")
        for dept in confidence_df['科室'].unique():
            dept_data = confidence_df[confidence_df['科室'] == dept]
            if len(dept_data) > 0:
                dept_ai_mean = dept_data['ai_avg_confidence'].mean()
                dept_no_ai_mean = dept_data['no_ai_avg_confidence'].mean()
                diff = dept_ai_mean - dept_no_ai_mean
                print(f"  {dept}: AI vs 无AI = {dept_ai_mean:.3f} vs {dept_no_ai_mean:.3f} (差异: {diff:+.3f}, n={len(dept_data)}对)")

if __name__ == "__main__":
    main()
