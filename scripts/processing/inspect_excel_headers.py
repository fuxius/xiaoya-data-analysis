#!/usr/bin/env python3
"""
检查Excel文件表头工具

此脚本用于检查两个Excel文件的表头结构，帮助了解数据结构
以便更好地进行数据合并。

作者: AICare研究团队
日期: 2025年9月
"""

import pandas as pd
from pathlib import Path
import sys

def inspect_excel_file(file_path):
    """
    检查Excel文件的结构
    
    参数:
        file_path (Path): Excel文件路径
    """
    print(f"\n{'='*60}")
    print(f"检查文件: {file_path.name}")
    print(f"{'='*60}")
    
    try:
        # 首先检查有多少个工作表
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"工作表数量: {len(sheet_names)}")
        print(f"工作表名称: {sheet_names}")
        
        # 检查每个工作表
        for i, sheet_name in enumerate(sheet_names):
            print(f"\n--- 工作表 {i+1}: '{sheet_name}' ---")
            
            # 读取工作表（只读取前几行以节省内存）
            df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
            
            print(f"数据形状: {df.shape}")
            print(f"列数: {len(df.columns)}")
            
            print("\n列名:")
            for j, col in enumerate(df.columns):
                print(f"  {j+1:2d}. {col}")
            
            print(f"\n前3行数据:")
            print(df.head(3).to_string())
            
            # 检查数据类型
            print(f"\n数据类型:")
            for col in df.columns:
                dtype = df[col].dtype
                non_null_count = df[col].count()
                print(f"  {col}: {dtype} (非空值: {non_null_count})")
                
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return False
    
    return True

def main():
    """主函数"""
    print("AICare CHI论文 - Excel文件表头检查工具")
    
    # 定义文件路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    print(f"基础目录: {base_dir}")
    print(f"数据目录: {data_dir}")
    print(f"数据目录是否存在: {data_dir.exists()}")
    
    files_to_check = [
        data_dir / '问卷收集情况.xlsx',
        data_dir / '问卷数据.xlsx'
    ]
    
    print(f"\n要检查的文件:")
    for i, file_path in enumerate(files_to_check, 1):
        print(f"  {i}. {file_path}")
        print(f"     存在: {file_path.exists()}")
        if file_path.exists():
            print(f"     大小: {file_path.stat().st_size} 字节")
    
    # 检查文件是否存在
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"错误: 文件不存在 - {file_path}")
            sys.exit(1)
    
    print(f"\n开始检查文件...")
    
    # 检查每个文件
    success_count = 0
    for file_path in files_to_check:
        try:
            if inspect_excel_file(file_path):
                success_count += 1
        except Exception as e:
            print(f"检查文件 {file_path} 时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"检查完成! 成功检查了 {success_count}/{len(files_to_check)} 个文件")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
