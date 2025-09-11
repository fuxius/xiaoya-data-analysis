#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取半结构化访谈数据中的定性洞察
"""

import pandas as pd
import json

def main():
    """提取访谈数据"""
    
    # 读取合并后的数据
    print("正在读取数据文件...")
    df = pd.read_excel('results/merged_dataset_simple.xlsx')
    
    # 定义参与者基本信息列
    participant_cols = [
        'ID',
        '科室',
        '7、您当前的身份或职称是？'
    ]
    
    # 定义访谈问题列（基于数据中的Q11-Q41列）
    interview_cols = [
        'Q11',  # 系统理解和困惑
        'Q12',  # 临床工作流程整合
        'Q13',  # 诊断效率和准确性
        'Q21',  # 信任建立时刻
        'Q22',  # 动态风险轨迹价值
        'Q23',  # 个体化指标分析交互
        'Q24',  # 人群级分析价值
        'Q25',  # LLM建议质量
        'Q26',  # AI与临床判断差异
        'Q31',  # 新发现的风险因素
        'Q41'   # 系统改进建议
    ]
    
    # 检查列是否存在
    missing_cols = []
    for col in participant_cols + interview_cols:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"警告：以下列不存在于数据中：")
        for col in missing_cols:
            print(f"  - {col}")
    
    # 提取存在的列
    available_cols = [col for col in participant_cols + interview_cols if col in df.columns]
    interview_data = df[available_cols].copy()
    
    # 重命名访谈问题列为更描述性的名称
    interview_rename_map = {
        'Q11': 'I01_系统理解困惑',
        'Q12': 'I02_临床工作整合',
        'Q13': 'I03_效率准确性',
        'Q21': 'I04_信任建立时刻',
        'Q22': 'I05_动态轨迹价值',
        'Q23': 'I06_个体化分析交互',
        'Q24': 'I07_人群分析价值',
        'Q25': 'I08_LLM建议质量',
        'Q26': 'I09_AI临床差异',
        'Q31': 'I10_新风险发现',
        'Q41': 'I11_改进建议'
    }
    
    # 应用重命名
    interview_data = interview_data.rename(columns=interview_rename_map)
    
    # 保存原始数据
    output_file = 'results/interview_raw_data.csv'
    interview_data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"访谈原始数据已保存到: {output_file}")
    
    # 显示数据概览
    print(f"\n数据概览:")
    print(f"参与者数量: {len(interview_data)}")
    print(f"数据列数: {len(interview_data.columns)}")
    
    # 统计各访谈问题的回答情况
    interview_columns = [col for col in interview_data.columns if col.startswith('I')]
    
    print(f"\n访谈问题回答情况:")
    for col in interview_columns:
        if col in interview_data.columns:
            non_null_count = interview_data[col].notna().sum()
            null_count = interview_data[col].isna().sum()
            print(f"{col}: 有效回答 {non_null_count}人, 无回答 {null_count}人")
    
    # 显示有访谈数据的参与者信息
    has_interview = interview_data[interview_columns].notna().any(axis=1)
    interviewed_participants = interview_data[has_interview]
    
    print(f"\n参与访谈的人员信息:")
    print(f"访谈参与者数量: {len(interviewed_participants)}")
    
    if '科室' in interviewed_participants.columns:
        print("访谈参与者科室分布:")
        print(interviewed_participants['科室'].value_counts())
    
    if '7、您当前的身份或职称是？' in interviewed_participants.columns:
        print("\n访谈参与者职称分布:")
        print(interviewed_participants['7、您当前的身份或职称是？'].value_counts())
    
    # 保存列名映射信息
    mapping_info = {
        'participant_columns': participant_cols,
        'interview_columns': interview_cols,
        'interview_rename_mapping': interview_rename_map,
        'interview_questions': {
            'I01_系统理解困惑': '您在使用系统过程中，有没有感到困惑或难以理解的地方？',
            'I02_临床工作整合': '您认为这个系统如何整合到您的日常临床工作流程中？',
            'I03_效率准确性': '您认为这个系统能否提升您的诊断效率和准确率？',
            'I04_信任建立时刻': '在使用过程中，什么时候您会觉得系统是可信的，什么时候会产生怀疑？',
            'I05_动态轨迹价值': '动态风险轨迹可视化对您理解病情有什么帮助？',
            'I06_个体化分析交互': '个体化关键指标分析的交互功能如何影响您对系统的认知？',
            'I07_人群分析价值': '人群级别的指标分析可视化对您有用吗？',
            'I08_LLM建议质量': '大语言模型给出的诊疗建议质量如何？',
            'I09_AI临床差异': '当AI的判断与您的临床判断不一致时，您如何处理？',
            'I10_新风险发现': '系统是否帮助您发现了之前可能忽略的风险因素？',
            'I11_改进建议': '您对系统有什么改进建议？'
        }
    }
    
    with open('results/interview_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n列名映射信息已保存到: results/interview_mapping.json")
    
    # 显示访谈数据示例（只显示有数据的列）
    print(f"\n访谈数据示例:")
    for col in interview_columns:
        if col in interview_data.columns:
            non_null_data = interview_data[interview_data[col].notna()]
            if len(non_null_data) > 0:
                print(f"\n{col} 示例回答:")
                print(f"  参与者 {non_null_data.iloc[0]['ID']}: {non_null_data.iloc[0][col][:100]}...")

if __name__ == "__main__":
    main()
