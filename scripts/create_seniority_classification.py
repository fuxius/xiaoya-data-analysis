#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建年资分类
"""

import pandas as pd
import json

def classify_seniority(title):
    """根据职称分类年资"""
    if pd.isna(title) or title == '(空)':
        return '未知'
    
    # 高年资：主任/副主任医师、主治医师
    high_seniority_keywords = ['主任医师', '副主任医师', '主治医师']
    
    for keyword in high_seniority_keywords:
        if keyword in str(title):
            return '高年资'
    
    # 其他为低年资
    return '低年资'

def main():
    """创建年资分类"""
    
    # 读取数据
    print("正在读取数据文件...")
    df = pd.read_excel('results/merged_dataset_simple.xlsx')
    
    # 检查职称列是否存在
    title_col = '7、您当前的身份或职称是？'
    if title_col not in df.columns:
        print(f"错误：未找到职称列 '{title_col}'")
        return
    
    # 创建年资分类
    df['年资分类'] = df[title_col].apply(classify_seniority)
    
    # 统计分类结果
    print("\n年资分类统计:")
    seniority_stats = df['年资分类'].value_counts()
    print(seniority_stats)
    
    print("\n详细分类结果:")
    for _, row in df[['ID', title_col, '年资分类']].iterrows():
        print(f"  {row['ID']}: {row[title_col]} -> {row['年资分类']}")
    
    # 创建分类映射
    classification_mapping = {}
    for _, row in df.iterrows():
        classification_mapping[row['ID']] = {
            'original_title': row[title_col],
            'seniority_level': row['年资分类']
        }
    
    # 保存分类结果
    output_file = 'results/seniority_classification.json'
    classification_data = {
        'classification_rules': {
            '高年资': ['主任医师', '副主任医师', '主治医师'],
            '低年资': ['其他职称（住院医师、护士等）']
        },
        'participant_classification': classification_mapping,
        'statistics': {
            '高年资人数': int(seniority_stats.get('高年资', 0)),
            '低年资人数': int(seniority_stats.get('低年资', 0)),
            '未知人数': int(seniority_stats.get('未知', 0)),
            '总人数': len(df)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(classification_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 年资分类结果已保存到: {output_file}")
    
    # 保存带年资分类的数据文件（可选）
    participant_data = df[['ID', '科室', title_col, '年资分类']].copy()
    participant_file = 'results/participants_with_seniority.csv'
    participant_data.to_csv(participant_file, index=False, encoding='utf-8-sig')
    print(f"✓ 参与者年资数据已保存到: {participant_file}")

if __name__ == "__main__":
    main()
