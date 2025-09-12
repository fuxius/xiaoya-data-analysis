#!/usr/bin/env python3
"""
分析患者基本信息表的详细统计
"""

import pandas as pd
import json

def main():
    # 读取基本信息表
    df = pd.read_excel("results/participants/participant_basic_info.xlsx")
    
    print("=== 患者基本信息表分析报告 ===\n")
    
    print(f"总参与者数: {len(df)}")
    print(f"总字段数: {len(df.columns)}\n")
    
    # 显示所有列名
    print("包含的字段:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    print()
    
    # 详细统计分析
    print("=== 详细统计分析 ===\n")
    
    # 1. 科室分布
    print("1. 科室分布:")
    dept_counts = df['科室'].value_counts()
    for dept, count in dept_counts.items():
        percentage = count / len(df) * 100
        print(f"   {dept}: {count}人 ({percentage:.1f}%)")
    print()
    
    # 2. 性别分布
    print("2. 性别分布:")
    gender_counts = df['3、性别'].value_counts()
    for gender, count in gender_counts.items():
        percentage = count / len(df) * 100
        print(f"   {gender}: {count}人 ({percentage:.1f}%)")
    print()
    
    # 3. 年龄段分布
    print("3. 年龄段分布:")
    age_counts = df['5、年龄段'].value_counts()
    for age, count in age_counts.items():
        percentage = count / len(df) * 100
        print(f"   {age}: {count}人 ({percentage:.1f}%)")
    print()
    
    # 4. 工作年限分布
    print("4. 临床工作年限分布:")
    exp_counts = df['6、您从事临床工作多少年了？'].value_counts()
    for exp, count in exp_counts.items():
        percentage = count / len(df) * 100
        print(f"   {exp}: {count}人 ({percentage:.1f}%)")
    print()
    
    # 5. 职称分布
    print("5. 职称分布:")
    title_counts = df['7、您当前的身份或职称是？'].value_counts()
    for title, count in title_counts.items():
        percentage = count / len(df) * 100
        print(f"   {title}: {count}人 ({percentage:.1f}%)")
    print()
    
    # 6. AI系统熟悉程度
    print("6. AI系统熟悉程度:")
    ai_fam_counts = df['8、您对人工智能赋能的临床决策支持系统的熟悉程度如何？1 = 完全不了解： 没听说过这个概念。2 = 略有耳闻： 听说过，但不清楚具体做什么。3 = 基本了解： 了解其大致定义和用途。4 = 比较了解： 对其应用和关键技术有较深理解。5 = 非常了解： 在该领域有专业知识或实践经验。'].value_counts()
    for level, count in ai_fam_counts.items():
        percentage = count / len(df) * 100
        print(f"   {level}: {count}人 ({percentage:.1f}%)")
    print()
    
    # 7. AI系统使用频率
    print("7. AI系统使用频率:")
    ai_freq_counts = df['9、在本次研究之前，您在临床工作中使用类似的人工智能决策支持系统的频率是？'].value_counts()
    for freq, count in ai_freq_counts.items():
        percentage = count / len(df) * 100
        print(f"   {freq}: {count}人 ({percentage:.1f}%)")
    print()
    
    # 8. 实验设计相关
    print("8. 实验设计分布:")
    print("   是否先使用AI分析系统:")
    ai_first_counts = df['是否先使用AI分析系统'].value_counts()
    for option, count in ai_first_counts.items():
        percentage = count / len(df) * 100
        print(f"     {option}: {count}人 ({percentage:.1f}%)")
    
    print("   是否参与访谈:")
    interview_counts = df['是否参与访谈'].value_counts()
    for option, count in interview_counts.items():
        percentage = count / len(df) * 100
        print(f"     {option}: {count}人 ({percentage:.1f}%)")
    
    print("   是否记录分析时间:")
    time_counts = df['是否记录分析时间'].value_counts()
    for option, count in time_counts.items():
        percentage = count / len(df) * 100
        print(f"     {option}: {count}人 ({percentage:.1f}%)")
    print()
    
    # 显示完整数据
    print("=== 完整数据预览 ===")
    print(df.to_string())

if __name__ == "__main__":
    main()
