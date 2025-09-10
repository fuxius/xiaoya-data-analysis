#!/usr/bin/env python3
"""
简单的问卷数据合并脚本

直接按ID列合并两个Excel文件，简单高效。

作者: AICare研究团队
日期: 2025年9月
"""

import pandas as pd
from pathlib import Path
import sys

def main():
    """主函数 - 简单直接的合并逻辑"""
    print("=== 简单问卷数据合并工具 ===")
    
    # 文件路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    results_dir = base_dir / 'results'
    
    collection_file = data_dir / '问卷收集情况.xlsx'
    responses_file = data_dir / '问卷数据.xlsx'
    output_file = results_dir / 'merged_dataset_simple.xlsx'
    
    print(f"读取文件1: {collection_file.name}")
    print(f"读取文件2: {responses_file.name}")
    
    try:
        # 读取两个Excel文件
        df_collection = pd.read_excel(collection_file)
        df_responses = pd.read_excel(responses_file)
        
        print(f"\n文件1形状: {df_collection.shape}")
        print(f"文件1的ID列: {df_collection['ID'].tolist()}")
        
        print(f"\n文件2形状: {df_responses.shape}")  
        print(f"文件2的ID列: {df_responses['ID'].tolist()}")
        
        # 按ID列进行合并 (左连接，以问卷数据为主)
        print(f"\n开始按ID列合并...")
        merged_df = pd.merge(
            df_responses,           # 左表：问卷数据（主要数据）
            df_collection,          # 右表：收集情况
            on='ID',               # 合并键
            how='left',            # 左连接
            suffixes=('', '_收集情况')  # 重复列的后缀
        )
        
        print(f"合并后形状: {merged_df.shape}")
        print(f"合并后的ID列: {merged_df['ID'].tolist()}")
        
        # 创建输出目录
        results_dir.mkdir(exist_ok=True)
        
        # 保存合并结果
        print(f"\n保存到: {output_file}")
        merged_df.to_excel(output_file, index=False)
        
        print(f"\n✅ 合并完成!")
        print(f"   - 输入文件1: {df_collection.shape[0]} 行")
        print(f"   - 输入文件2: {df_responses.shape[0]} 行") 
        print(f"   - 合并结果: {merged_df.shape[0]} 行, {merged_df.shape[1]} 列")
        print(f"   - 输出文件: {output_file}")
        
        # 显示一些基本信息
        print(f"\n📊 合并后的列数: {len(merged_df.columns)}")
        print(f"📋 前几列名称: {list(merged_df.columns[:5])}")
        
    except Exception as e:
        print(f"❌ 合并过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
