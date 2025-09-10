#!/usr/bin/env python3
"""
表头提取和持久化工具

读取整理后的merged_dataset_simple.xlsx文件的表头结构，
并将其保存为JSON文件，方便后续数据分析使用。

作者: AICare研究团队
日期: 2025年9月
"""

import pandas as pd
import json
from pathlib import Path
import sys
from datetime import datetime

def extract_headers_info(excel_file):
    """
    提取Excel文件的表头信息
    
    参数:
        excel_file (Path): Excel文件路径
    
    返回:
        dict: 包含表头信息的字典
    """
    print(f"正在读取文件: {excel_file}")
    
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_file)
        
        print(f"数据形状: {df.shape}")
        print(f"总列数: {len(df.columns)}")
        
        # 构建表头信息字典
        headers_info = {
            "metadata": {
                "file_name": excel_file.name,
                "extraction_date": datetime.now().isoformat(),
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "description": "AICare CHI论文问卷数据表头信息"
            },
            "columns": []
        }
        
        # 遍历每一列，收集详细信息
        for i, col_name in enumerate(df.columns):
            col_info = {
                "index": i,
                "name": col_name,
                "data_type": str(df[col_name].dtype),
                "non_null_count": int(df[col_name].count()),
                "null_count": int(df[col_name].isnull().sum()),
                "unique_values_count": int(df[col_name].nunique())
            }
            
            # 如果是数值型数据，添加统计信息
            if df[col_name].dtype in ['int64', 'float64']:
                col_info["statistics"] = {
                    "min": float(df[col_name].min()) if not df[col_name].empty else None,
                    "max": float(df[col_name].max()) if not df[col_name].empty else None,
                    "mean": float(df[col_name].mean()) if not df[col_name].empty else None
                }
            
            # 如果是分类数据且唯一值不多，添加唯一值列表
            if df[col_name].nunique() <= 20 and df[col_name].nunique() > 0:
                unique_vals = df[col_name].dropna().unique().tolist()
                # 转换numpy类型为Python原生类型
                col_info["unique_values"] = [str(val) for val in unique_vals]
            
            headers_info["columns"].append(col_info)
        
        return headers_info
        
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

def categorize_columns(headers_info):
    """
    根据列名对列进行分类
    
    参数:
        headers_info (dict): 表头信息字典
    
    返回:
        dict: 分类后的列信息
    """
    categories = {
        "participant_info": [],      # 参与者基本信息
        "risk_assessment": [],       # 风险评估相关
        "nasa_tlx": [],             # NASA-TLX工作负荷
        "sus_scale": [],            # 系统易用性量表
        "trust_scale": [],          # 自动化信任量表
        "system_feedback": [],      # 系统功能反馈
        "collection_metadata": [],  # 收集情况元数据
        "other": []                 # 其他
    }
    
    for col in headers_info["columns"]:
        col_name = col["name"].lower()
        
        # 参与者基本信息
        if any(keyword in col_name for keyword in ["id", "性别", "年龄", "职称", "工作年限", "临床专业", "熟悉程度"]):
            categories["participant_info"].append(col)
        # NASA-TLX相关
        elif any(keyword in col_name for keyword in ["脑力需求", "体力需求", "时间压力", "任务表现", "努力程度", "挫败感", "mental demand", "physical demand"]):
            categories["nasa_tlx"].append(col)
        # 系统易用性量表
        elif "系统易用性量表" in col_name or "sus" in col_name:
            categories["sus_scale"].append(col)
        # 自动化信任量表
        elif "自动化信任量表" in col_name or "trust" in col_name:
            categories["trust_scale"].append(col)
        # 系统功能反馈
        elif any(keyword in col_name for keyword in ["功能反馈", "动态风险", "可视化", "大语言模型"]):
            categories["system_feedback"].append(col)
        # 风险评估相关
        elif any(keyword in col_name for keyword in ["评估", "风险", "信心", "临床指标", "患者", "孕妇"]):
            categories["risk_assessment"].append(col)
        # 收集情况元数据
        elif any(keyword in col_name for keyword in ["时间", "访谈", "记录", "备注", "收集情况"]):
            categories["collection_metadata"].append(col)
        else:
            categories["other"].append(col)
    
    return categories

def main():
    """主函数"""
    print("=== AICare 表头提取和持久化工具 ===")
    
    # 文件路径
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results'
    
    excel_file = results_dir / 'merged_dataset_simple.xlsx'
    json_file = results_dir / 'dataset_headers.json'
    categorized_json_file = results_dir / 'dataset_headers_categorized.json'
    
    print(f"输入文件: {excel_file}")
    print(f"输出文件1: {json_file}")
    print(f"输出文件2: {categorized_json_file}")
    
    # 检查输入文件是否存在
    if not excel_file.exists():
        print(f"❌ 错误: 文件不存在 - {excel_file}")
        print("请确保已经运行过合并脚本并生成了merged_dataset_simple.xlsx文件")
        sys.exit(1)
    
    # 提取表头信息
    print(f"\n📋 开始提取表头信息...")
    headers_info = extract_headers_info(excel_file)
    
    if headers_info is None:
        print("❌ 提取表头信息失败")
        sys.exit(1)
    
    # 保存基础表头信息
    print(f"\n💾 保存基础表头信息到: {json_file}")
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(headers_info, f, ensure_ascii=False, indent=2)
        print("✅ 基础表头信息保存成功")
    except Exception as e:
        print(f"❌ 保存基础表头信息失败: {str(e)}")
        sys.exit(1)
    
    # 对列进行分类
    print(f"\n🏷️  对列进行分类...")
    categorized_info = {
        "metadata": headers_info["metadata"],
        "categories": categorize_columns(headers_info),
        "summary": {}
    }
    
    # 添加分类汇总信息
    for category, columns in categorized_info["categories"].items():
        categorized_info["summary"][category] = {
            "count": len(columns),
            "column_names": [col["name"] for col in columns]
        }
    
    # 保存分类后的表头信息
    print(f"\n💾 保存分类表头信息到: {categorized_json_file}")
    try:
        with open(categorized_json_file, 'w', encoding='utf-8') as f:
            json.dump(categorized_info, f, ensure_ascii=False, indent=2)
        print("✅ 分类表头信息保存成功")
    except Exception as e:
        print(f"❌ 保存分类表头信息失败: {str(e)}")
        sys.exit(1)
    
    # 打印汇总信息
    print(f"\n📊 表头提取完成!")
    print(f"   - 总列数: {headers_info['metadata']['total_columns']}")
    print(f"   - 总行数: {headers_info['metadata']['total_rows']}")
    print(f"\n🏷️  列分类汇总:")
    for category, info in categorized_info["summary"].items():
        if info["count"] > 0:
            print(f"   - {category}: {info['count']} 列")
    
    print(f"\n📁 输出文件:")
    print(f"   - 基础表头: {json_file}")
    print(f"   - 分类表头: {categorized_json_file}")

if __name__ == "__main__":
    main()
