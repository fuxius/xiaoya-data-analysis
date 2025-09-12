#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ2分析：诊断效率分析
比较AI辅助组与无辅助组的案例评估时间，贯穿不同群体差异分析

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

def extract_case_times(df):
    """提取案例时间数据 - 被试内设计"""
    print("\n正在提取案例时间数据...")
    
    # 找到所有案例时间列
    time_cols = [col for col in df.columns if 'case' in col and 'time' in col]
    time_cols = sorted(time_cols)  # 确保顺序正确
    print(f"找到案例时间列: {len(time_cols)}个")
    print(f"案例时间列: {time_cols}")
    
    # 为每个参与者提取时间数据（宽格式）
    efficiency_data = []
    
    for idx, row in df.iterrows():
        # 根据条件分配规则提取时间数据
        ai_times = []
        no_ai_times = []
        
        if row['ai_first']:
            # 先用AI: case01-05是AI辅助，case06-10是无辅助
            ai_cols = [col for col in time_cols if any(f'case0{i}_time' in col for i in range(1, 6))]
            no_ai_cols = [col for col in time_cols if any(f'case{i:02d}_time' in col for i in range(6, 11))]
        else:
            # 后用AI: case01-05是无辅助，case06-10是AI辅助
            no_ai_cols = [col for col in time_cols if any(f'case0{i}_time' in col for i in range(1, 6))]
            ai_cols = [col for col in time_cols if any(f'case{i:02d}_time' in col for i in range(6, 11))]
        
        # 提取AI辅助条件的时间
        for col in ai_cols:
            if col in row and pd.notna(row[col]):
                ai_times.append(row[col])
        
        # 提取无辅助条件的时间
        for col in no_ai_cols:
            if col in row and pd.notna(row[col]):
                no_ai_times.append(row[col])
        
        # 只有当两种条件都有数据时才包含此参与者
        if len(ai_times) > 0 and len(no_ai_times) > 0:
            efficiency_data.append({
                'ID': row['ID'],
                '科室': row['科室'],
                '年资分类': row['年资分类'],
                'hospital': row['hospital'],
                'ai_first': row['ai_first'],
                'ai_case_count': len(ai_times),
                'no_ai_case_count': len(no_ai_times),
                'ai_avg_time': np.mean(ai_times),
                'no_ai_avg_time': np.mean(no_ai_times),
                'ai_total_time': sum(ai_times),
                'no_ai_total_time': sum(no_ai_times),
                'time_difference': np.mean(ai_times) - np.mean(no_ai_times)
            })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    print(f"有效参与者数据: {len(efficiency_df)}个（具有配对数据）")
    
    return efficiency_df

def create_long_format_data(efficiency_df):
    """创建长格式数据用于分组分析"""
    print("\n正在创建长格式数据...")
    
    long_format_data = []
    
    for idx, row in efficiency_df.iterrows():
        # AI辅助条件
        ai_data = {
            'ID': row['ID'],
            '科室': row['科室'],
            '年资分类': row['年资分类'],
            'hospital': row['hospital'],
            'ai_first': row['ai_first'],
            'condition_type': 'AI辅助组',
            'avg_time_seconds': row['ai_avg_time'],
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
            'avg_time_seconds': row['no_ai_avg_time'],
            'case_count': row['no_ai_case_count']
        }
        long_format_data.append(no_ai_data)
    
    efficiency_long_df = pd.DataFrame(long_format_data)
    print(f"长格式数据记录数: {len(efficiency_long_df)}个")
    
    return efficiency_long_df

def calculate_efficiency_statistics(efficiency_df):
    """计算诊断效率统计分析 - 使用rm-ANOVA替代配对t检验"""
    print("\n=== 诊断效率统计分析 (rm-ANOVA) ===")
    
    stats_data = []
    
    # 准备长格式数据用于rm-ANOVA
    long_format_data = []
    for idx, row in efficiency_df.iterrows():
        # AI辅助条件
        long_format_data.append({
            'participant_id': row['ID'],
            'condition': 'AI辅助',
            'avg_time': row['ai_avg_time'],
            'seniority': row['年资分类'],
            'department': row['科室'],
            'hospital': row['hospital']
        })
        
        # 无辅助条件
        long_format_data.append({
            'participant_id': row['ID'],
            'condition': '无辅助',
            'avg_time': row['no_ai_avg_time'],
            'seniority': row['年资分类'],
            'department': row['科室'],
            'hospital': row['hospital']
        })
    
    efficiency_long_df = pd.DataFrame(long_format_data)
    
    # 1. 总体统计
    print("\n1. 总体效率统计:")
    
    # 描述性统计
    ai_data = efficiency_long_df[efficiency_long_df['condition'] == 'AI辅助']
    no_ai_data = efficiency_long_df[efficiency_long_df['condition'] == '无辅助']
    
    for condition, data in [('AI辅助组', ai_data), ('无辅助组', no_ai_data)]:
        times = data['avg_time']
        stats_data.append({
            '分析维度': '总体',
            '分组': condition,
            '样本数': len(times),
            '平均时间_秒': round(times.mean(), 2),
            '标准差_秒': round(times.std(), 2),
            '中位数_秒': round(times.median(), 2),
            '最小值_秒': round(times.min(), 2),
            '最大值_秒': round(times.max(), 2),
            '平均时间_分钟': round(times.mean() / 60, 2)
        })
        print(f"  {condition}: n={len(times)}, M={times.mean():.2f}s ({times.mean()/60:.2f}min), SD={times.std():.2f}s")
    
    # 执行rm-ANOVA
    print(f"\n执行总体rm-ANOVA分析:")
    rm_results = perform_rm_anova_analysis(
        efficiency_long_df,
        participant_col='participant_id',
        condition_col='condition',
        dv_col='avg_time'
    )
    
    if rm_results:
        print_rm_anova_summary(rm_results, "总体效率分析")
        
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
    print("\n2. 按科室效率统计:")
    for dept in efficiency_df['科室'].unique():
        dept_long_data = efficiency_long_df[efficiency_long_df['department'] == dept]
        
        if len(dept_long_data) < 4:  # 至少需要2个参与者×2个条件
            print(f"  {dept}: 数据不足，跳过分析")
            continue
            
        print(f"\n  {dept}:")
        
        # 描述性统计
        for condition in ['AI辅助', '无辅助']:
            condition_data = dept_long_data[dept_long_data['condition'] == condition]
            times = condition_data['avg_time']
            
            condition_label = f'{condition}组'
            stats_data.append({
                '分析维度': f'科室_{dept}',
                '分组': condition_label,
                '样本数': len(times),
                '平均时间_秒': round(times.mean(), 2),
                '标准差_秒': round(times.std(), 2),
                '中位数_秒': round(times.median(), 2),
                '最小值_秒': round(times.min(), 2),
                '最大值_秒': round(times.max(), 2),
                '平均时间_分钟': round(times.mean() / 60, 2)
            })
            print(f"    {condition_label}: n={len(times)}, M={times.mean():.2f}s ({times.mean()/60:.2f}min)")
        
        # 科室内rm-ANOVA
        print(f"    执行{dept}科室rm-ANOVA:")
        dept_rm_results = perform_rm_anova_analysis(
            dept_long_data,
            participant_col='participant_id',
            condition_col='condition',
            dv_col='avg_time'
        )
        
        if dept_rm_results:
            print_rm_anova_summary(dept_rm_results, f"{dept}科室效率分析")
            
            # 提取统计量并添加到结果中
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
    print("\n3. 按年资效率统计:")
    for seniority in efficiency_df['年资分类'].unique():
        if seniority == '未知':
            continue
            
        seniority_data = efficiency_df[efficiency_df['年资分类'] == seniority]
        if len(seniority_data) == 0:
            continue
            
        print(f"\n  {seniority}:")
        
        # AI辅助组统计
        seniority_ai_times = seniority_data['ai_avg_time']
        stats_data.append({
            '分析维度': f'年资_{seniority}',
            '分组': 'AI辅助组',
            '样本数': len(seniority_ai_times),
            '平均时间_秒': round(seniority_ai_times.mean(), 2),
            '标准差_秒': round(seniority_ai_times.std(), 2),
            '中位数_秒': round(seniority_ai_times.median(), 2),
            '最小值_秒': round(seniority_ai_times.min(), 2),
            '最大值_秒': round(seniority_ai_times.max(), 2),
            '平均时间_分钟': round(seniority_ai_times.mean() / 60, 2)
        })
        print(f"    AI辅助组: n={len(seniority_ai_times)}, M={seniority_ai_times.mean():.2f}s")
        
        # 无辅助组统计
        seniority_no_ai_times = seniority_data['no_ai_avg_time']
        stats_data.append({
            '分析维度': f'年资_{seniority}',
            '分组': '无辅助组',
            '样本数': len(seniority_no_ai_times),
            '平均时间_秒': round(seniority_no_ai_times.mean(), 2),
            '标准差_秒': round(seniority_no_ai_times.std(), 2),
            '中位数_秒': round(seniority_no_ai_times.median(), 2),
            '最小值_秒': round(seniority_no_ai_times.min(), 2),
            '最大值_秒': round(seniority_no_ai_times.max(), 2),
            '平均时间_分钟': round(seniority_no_ai_times.mean() / 60, 2)
        })
        print(f"    无辅助组: n={len(seniority_no_ai_times)}, M={seniority_no_ai_times.mean():.2f}s")
        
        # 年资内配对t检验
        if len(seniority_data) > 1:
            try:
                t_stat, p_value = stats.ttest_rel(seniority_ai_times, seniority_no_ai_times)
                effect_size = calculate_cohens_d_paired(seniority_ai_times, seniority_no_ai_times)
                print(f"    {seniority}年资配对t检验: t={t_stat:.4f}, p={p_value:.4f}")
                
                # 添加检验结果
                seniority_stats = [s for s in stats_data if s['分析维度'] == f'年资_{seniority}']
                for stat in seniority_stats:
                    stat['t_value'] = t_stat
                    stat['p_value'] = p_value
                    stat['effect_size'] = effect_size
            except:
                print(f"    {seniority}年资t检验: 计算失败")
    
    # 4. 按医院统计
    print("\n4. 按医院效率统计:")
    for hospital in efficiency_df['hospital'].unique():
        hospital_data = efficiency_df[efficiency_df['hospital'] == hospital]
        if len(hospital_data) == 0:
            continue
            
        print(f"\n  {hospital}医院:")
        
        # AI辅助组统计
        hospital_ai_times = hospital_data['ai_avg_time']
        stats_data.append({
            '分析维度': f'医院_{hospital}',
            '分组': 'AI辅助组',
            '样本数': len(hospital_ai_times),
            '平均时间_秒': round(hospital_ai_times.mean(), 2),
            '标准差_秒': round(hospital_ai_times.std(), 2),
            '中位数_秒': round(hospital_ai_times.median(), 2),
            '最小值_秒': round(hospital_ai_times.min(), 2),
            '最大值_秒': round(hospital_ai_times.max(), 2),
            '平均时间_分钟': round(hospital_ai_times.mean() / 60, 2)
        })
        print(f"    AI辅助组: n={len(hospital_ai_times)}, M={hospital_ai_times.mean():.2f}s")
        
        # 无辅助组统计
        hospital_no_ai_times = hospital_data['no_ai_avg_time']
        stats_data.append({
            '分析维度': f'医院_{hospital}',
            '分组': '无辅助组',
            '样本数': len(hospital_no_ai_times),
            '平均时间_秒': round(hospital_no_ai_times.mean(), 2),
            '标准差_秒': round(hospital_no_ai_times.std(), 2),
            '中位数_秒': round(hospital_no_ai_times.median(), 2),
            '最小值_秒': round(hospital_no_ai_times.min(), 2),
            '最大值_秒': round(hospital_no_ai_times.max(), 2),
            '平均时间_分钟': round(hospital_no_ai_times.mean() / 60, 2)
        })
        print(f"    无辅助组: n={len(hospital_no_ai_times)}, M={hospital_no_ai_times.mean():.2f}s")
        
        # 医院内配对t检验
        if len(hospital_data) > 1:
            try:
                t_stat, p_value = stats.ttest_rel(hospital_ai_times, hospital_no_ai_times)
                effect_size = calculate_cohens_d_paired(hospital_ai_times, hospital_no_ai_times)
                print(f"    {hospital}医院配对t检验: t={t_stat:.4f}, p={p_value:.4f}")
                
                # 添加检验结果
                hospital_stats = [s for s in stats_data if s['分析维度'] == f'医院_{hospital}']
                for stat in hospital_stats:
                    stat['t_value'] = t_stat
                    stat['p_value'] = p_value
                    stat['effect_size'] = effect_size
            except:
                print(f"    {hospital}医院t检验: 计算失败")
    
    return pd.DataFrame(stats_data)

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

def main():
    """主函数"""
    print("=== RQ2: 诊断效率分析 (被试内设计) ===")
    
    # 加载数据并创建分组
    df = load_data_and_create_groups()
    
    # 提取案例时间数据（配对数据）
    efficiency_df = extract_case_times(df)
    
    # 创建长格式数据用于分组分析
    efficiency_long_df = create_long_format_data(efficiency_df)
    
    # 计算统计分析
    efficiency_stats = calculate_efficiency_statistics(efficiency_df)
    
    # 保存结果文件
    
    # 1. 原始效率数据（宽格式，配对数据）
    raw_file = 'results/efficiency/efficiency_raw_data.csv'
    efficiency_df.to_csv(raw_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 效率原始数据（配对）已保存到: {raw_file}")
    
    # 2. 长格式效率数据
    long_file = 'results/efficiency/efficiency_long_data.csv'
    efficiency_long_df.to_csv(long_file, index=False, encoding='utf-8-sig')
    print(f"✓ 效率长格式数据已保存到: {long_file}")
    
    # 3. 效率统计分析
    stats_file = 'results/efficiency/efficiency_analysis.csv'
    efficiency_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"✓ 效率统计分析已保存到: {stats_file}")
    
    # 显示关键结果概览
    print(f"\n=== 诊断效率分析结果概览 ===")
    print(f"有效配对参与者数量: {len(efficiency_df)}")
    
    # 总体效果（配对数据）
    if len(efficiency_df) > 0:
        ai_mean = efficiency_df['ai_avg_time'].mean()
        no_ai_mean = efficiency_df['no_ai_avg_time'].mean()
        time_diff = ai_mean - no_ai_mean
        
        print(f"\n总体效果（配对分析）:")
        print(f"  AI辅助组平均时间: {ai_mean:.2f}秒 ({ai_mean/60:.2f}分钟)")
        print(f"  无辅助组平均时间: {no_ai_mean:.2f}秒 ({no_ai_mean/60:.2f}分钟)")
        print(f"  时间差异: {time_diff:.2f}秒 ({'增加' if time_diff > 0 else '减少'}{abs(time_diff)/60:.2f}分钟)")
        
        # 效果解释
        if abs(time_diff) < 10:
            effect_desc = "无明显差异"
        elif time_diff < 0:
            effect_desc = "AI辅助提升效率"
        else:
            effect_desc = "AI辅助降低效率"
        
        print(f"  效果评价: {effect_desc}")
        
        # rm-ANOVA结果已在统计分析中显示
        print(f"  rm-ANOVA结果详见上方统计分析")
    
    # 按科室效果（配对分析）
    if '科室' in efficiency_df.columns:
        print(f"\n按科室效果（配对分析）:")
        for dept in efficiency_df['科室'].unique():
            dept_data = efficiency_df[efficiency_df['科室'] == dept]
            if len(dept_data) > 0:
                dept_ai_mean = dept_data['ai_avg_time'].mean()
                dept_no_ai_mean = dept_data['no_ai_avg_time'].mean()
                diff = dept_ai_mean - dept_no_ai_mean
                print(f"  {dept}: AI vs 无AI = {dept_ai_mean:.1f}s vs {dept_no_ai_mean:.1f}s (差异: {diff:+.1f}s, n={len(dept_data)}对)")

if __name__ == "__main__":
    main()
