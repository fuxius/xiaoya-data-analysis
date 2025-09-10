#!/usr/bin/env python3
"""
提取患者基本信息表（CSV格式）
从merged_dataset_simple.xlsx中提取前几列的患者基本信息，
一直到"9、在本次研究之前，您在临床工作中使用类似的人工智能决策支持系统的频率是？"
输出为CSV格式
"""

import pandas as pd
import json
from pathlib import Path

def main():
    # 读取原始数据
    input_file = "results/merged_dataset_simple.xlsx"
    output_file = "results/participant_basic_info.csv"
    
    print(f"正在读取文件: {input_file}")
    
    try:
        # 读取Excel文件
        df = pd.read_excel(input_file)
        
        # 提取前11列（索引0-10）
        target_columns = df.columns[:11]
        
        print(f"提取的列数: {len(target_columns)}")
        print("提取的列名:")
        for i, col in enumerate(target_columns):
            print(f"  {i}: {col}")
        
        # 提取基本信息列
        basic_info_df = df[target_columns].copy()
        
        # 保存为CSV文件（UTF-8编码，确保中文正确显示）
        basic_info_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n成功创建患者基本信息表（CSV格式）: {output_file}")
        print(f"数据行数: {len(basic_info_df)}")
        print(f"数据列数: {len(basic_info_df.columns)}")
        
        # 显示数据预览
        print("\n数据预览:")
        print(basic_info_df.head())
        
        # 生成简单的统计信息
        print(f"\n基本统计信息:")
        print(f"- 参与者总数: {len(basic_info_df)}")
        print(f"- 科室分布:")
        if '科室' in basic_info_df.columns:
            dept_counts = basic_info_df['科室'].value_counts()
            for dept, count in dept_counts.items():
                print(f"  {dept}: {count}人")
        
        print(f"- 性别分布:")
        if '3、性别' in basic_info_df.columns:
            gender_counts = basic_info_df['3、性别'].value_counts()
            for gender, count in gender_counts.items():
                print(f"  {gender}: {count}人")
        
        print(f"\n文件已保存为CSV格式，使用UTF-8编码，确保中文字符正确显示。")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
