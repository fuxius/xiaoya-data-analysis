#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取AICare系统功能有用性评价数据
"""

import pandas as pd
import json

def main():
    """提取功能有用性数据"""
    
    # 读取合并后的数据
    print("正在读取数据文件...")
    df = pd.read_excel('results/merged_dataset_simple.xlsx')
    
    # 定义参与者基本信息列
    participant_cols = [
        'ID',
        '科室',
        '7、您当前的身份或职称是？'
    ]
    
    # 定义四个核心功能模块的评价列
    feature_cols = [
        '74、"小雅医生"系统功能反馈请评价"小雅医生"系统的各项功能对您的诊断过程有多大帮助。—动态风险轨迹可视化',
        '74、可交互的个体化关键指标列表',
        '74、人群层面的指标分析可视化',
        '74、大语言模型驱动的诊疗建议'
    ]
    
    # 检查列是否存在
    missing_cols = []
    for col in participant_cols + feature_cols:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"警告：以下列不存在于数据中：")
        for col in missing_cols:
            print(f"  - {col}")
    
    # 提取存在的列
    available_cols = [col for col in participant_cols + feature_cols if col in df.columns]
    feature_data = df[available_cols].copy()
    
    # 重命名功能评价列为简洁的名称
    feature_rename_map = {
        '74、"小雅医生"系统功能反馈请评价"小雅医生"系统的各项功能对您的诊断过程有多大帮助。—动态风险轨迹可视化': 'F1_动态风险轨迹可视化',
        '74、可交互的个体化关键指标列表': 'F2_个体化关键指标分析',
        '74、人群层面的指标分析可视化': 'F3_人群级别指标分析',
        '74、大语言模型驱动的诊疗建议': 'F4_LLM诊疗建议'
    }
    
    # 应用重命名
    feature_data = feature_data.rename(columns=feature_rename_map)
    
    # 保存原始数据
    output_file = 'results/feature_usefulness_raw_data.csv'
    feature_data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"功能有用性原始数据已保存到: {output_file}")
    
    # 显示数据概览
    print(f"\n数据概览:")
    print(f"参与者数量: {len(feature_data)}")
    print(f"数据列数: {len(feature_data.columns)}")
    
    # 统计各功能模块的评价分布
    feature_columns = [col for col in feature_data.columns if col.startswith('F')]
    
    print(f"\n各功能模块评价分布:")
    for col in feature_columns:
        if col in feature_data.columns:
            print(f"\n{col}:")
            value_counts = feature_data[col].value_counts()
            for value, count in value_counts.items():
                percentage = count / len(feature_data) * 100
                print(f"  {value}: {count}人 ({percentage:.1f}%)")
    
    # 显示数据示例
    print(f"\n功能有用性数据示例（前5行）:")
    print(feature_data.head())
    
    # 保存列名映射信息
    mapping_info = {
        'participant_columns': participant_cols,
        'feature_columns': feature_cols,
        'feature_rename_mapping': feature_rename_map,
        'feature_descriptions': {
            'F1_动态风险轨迹可视化': '显示患者每次就诊时风险预测的折线图',
            'F2_个体化关键指标分析': '每次就诊的特征重要性分析，支持交互式探索',
            'F3_人群级别指标分析': '数据集中特征重要性、数值和患者风险的3D和2D可视化',
            'F4_LLM诊疗建议': '基于患者EHR数据、风险轨迹和临床指标，使用DeepSeek-V3.1提供分析和建议'
        },
        'rating_scale': ['极其有用', '比较有用', '一般有用', '有点没用', '完全没用']
    }
    
    with open('results/feature_usefulness_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n列名映射信息已保存到: results/feature_usefulness_mapping.json")

if __name__ == "__main__":
    main()
