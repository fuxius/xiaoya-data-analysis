#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化信任度量表分析可视化
生成信任度分析的图表和报告

作者: AICare研究团队
日期: 2025年9月
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_plotting_style():
    """设置绘图样式"""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 设置图形参数
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })

def load_trust_data():
    """加载信任度分析数据"""
    print("正在加载信任度分析数据...")
    
    # 加载原始数据
    trust_data = pd.read_csv('results/trust_scale_data.csv')
    
    # 加载统计分析结果
    trust_stats = pd.read_csv('results/trust_scale_analysis.csv')
    
    # 加载维度分析结果
    trust_dimensions = pd.read_csv('results/trust_scale_dimensions.csv')
    
    return trust_data, trust_stats, trust_dimensions

def plot_trust_overview(trust_stats, save_path='results/trust_scale_overview.png'):
    """绘制信任度总览图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('自动化信任度量表分析总览', fontsize=18, fontweight='bold')
    
    # 1. 总体信任度分布
    overall_data = trust_stats[trust_stats['分析维度'] == '总体']
    if not overall_data.empty:
        ax1.bar(['总体信任度'], overall_data['平均信任度'], 
                color='skyblue', alpha=0.7, width=0.5)
        ax1.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='中等信任阈值')
        ax1.axhline(y=5.5, color='green', linestyle='--', alpha=0.5, label='高信任阈值')
        ax1.set_ylabel('平均信任度 (1-7分)')
        ax1.set_title('总体信任度水平')
        ax1.set_ylim(1, 7)
        ax1.legend()
        
        # 添加数值标签
        for i, v in enumerate(overall_data['平均信任度']):
            ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 按科室信任度比较
    dept_data = trust_stats[trust_stats['分析维度'].str.contains('科室_')]
    if not dept_data.empty:
        dept_names = [name.replace('科室_', '') for name in dept_data['分组']]
        dept_scores = dept_data['平均信任度']
        
        bars = ax2.bar(dept_names, dept_scores, 
                      color=['lightcoral', 'lightgreen', 'lightskyblue'], alpha=0.7)
        ax2.axhline(y=4, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=5.5, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel('平均信任度 (1-7分)')
        ax2.set_title('按科室信任度比较')
        ax2.set_ylim(1, 7)
        ax2.tick_params(axis='x', rotation=15)
        
        # 添加数值标签
        for bar, score in zip(bars, dept_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, score + 0.1, 
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 按年资信任度比较
    seniority_data = trust_stats[trust_stats['分析维度'].str.contains('年资_')]
    if not seniority_data.empty:
        seniority_names = seniority_data['分组']
        seniority_scores = seniority_data['平均信任度']
        
        bars = ax3.bar(seniority_names, seniority_scores, 
                      color=['orange', 'purple'], alpha=0.7)
        ax3.axhline(y=4, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=5.5, color='green', linestyle='--', alpha=0.5)
        ax3.set_ylabel('平均信任度 (1-7分)')
        ax3.set_title('按年资信任度比较')
        ax3.set_ylim(1, 7)
        
        # 添加数值标签
        for bar, score in zip(bars, seniority_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, score + 0.1, 
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. AI使用顺序信任度比较
    ai_data = trust_stats[trust_stats['分析维度'] == 'AI使用顺序']
    if not ai_data.empty:
        ai_names = ai_data['分组']
        ai_scores = ai_data['平均信任度']
        
        bars = ax4.bar(ai_names, ai_scores, 
                      color=['lightblue', 'lightpink'], alpha=0.7)
        ax4.axhline(y=4, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=5.5, color='green', linestyle='--', alpha=0.5)
        ax4.set_ylabel('平均信任度 (1-7分)')
        ax4.set_title('按AI使用顺序信任度比较')
        ax4.set_ylim(1, 7)
        
        # 添加数值标签
        for bar, score in zip(bars, ai_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, score + 0.1, 
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 添加统计显著性信息
        if not pd.isna(ai_data['p_value'].iloc[0]):
            p_val = ai_data['p_value'].iloc[0]
            ax4.text(0.5, 6.5, f'p = {p_val:.3f}', ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ 信任度总览图已保存到: {save_path}")

def plot_trust_dimensions(trust_dimensions, save_path='results/trust_scale_dimensions.png'):
    """绘制信任度各维度分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('自动化信任度量表维度分析', fontsize=16, fontweight='bold')
    
    # 准备数据
    dimensions = trust_dimensions['信任维度'].tolist()
    scores = trust_dimensions['平均分'].tolist()
    std_devs = trust_dimensions['标准差'].tolist()
    
    # 简化维度名称以便显示
    short_names = []
    for dim in dimensions:
        if len(dim) > 15:
            short_names.append(dim[:15] + '...')
        else:
            short_names.append(dim)
    
    # 1. 维度平均分对比
    bars = ax1.barh(short_names, scores, color=plt.cm.viridis(np.linspace(0, 1, len(scores))))
    ax1.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='中等信任阈值')
    ax1.axvline(x=5.5, color='green', linestyle='--', alpha=0.5, label='高信任阈值')
    ax1.set_xlabel('平均分 (1-7分)')
    ax1.set_title('各维度信任度平均分')
    ax1.set_xlim(1, 7)
    ax1.legend()
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax1.text(score + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', ha='left', va='center', fontweight='bold')
    
    # 2. 维度得分分布（带误差线）
    y_pos = np.arange(len(dimensions))
    ax2.barh(y_pos, scores, xerr=std_devs, 
             color=plt.cm.plasma(np.linspace(0, 1, len(scores))), 
             alpha=0.7, capsize=5)
    ax2.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='中等信任阈值')
    ax2.axvline(x=5.5, color='green', linestyle='--', alpha=0.5, label='高信任阈值')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_names)
    ax2.set_xlabel('平均分 ± 标准差')
    ax2.set_title('各维度信任度分布')
    ax2.set_xlim(1, 7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ 信任度维度分析图已保存到: {save_path}")

def plot_trust_correlation_heatmap(trust_data, save_path='results/trust_scale_correlation.png'):
    """绘制信任度维度相关性热图"""
    # 提取信任度详细维度数据
    trust_detail_cols = [col for col in trust_data.columns 
                        if col.startswith('trust_') and col not in 
                        ['trust_total_score', 'trust_avg_score', 'trust_valid_items']]
    
    if len(trust_detail_cols) < 2:
        print("⚠️ 信任度维度数据不足，跳过相关性分析")
        return
    
    # 计算相关性矩阵
    trust_corr = trust_data[trust_detail_cols].corr()
    
    # 简化列名
    simplified_names = []
    for col in trust_detail_cols:
        name = col.replace('trust_', '').split('_', 1)[1] if '_' in col[6:] else col
        if len(name) > 20:
            name = name[:20] + '...'
        simplified_names.append(name)
    
    trust_corr.index = simplified_names
    trust_corr.columns = simplified_names
    
    # 绘制热图
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(trust_corr, dtype=bool))
    sns.heatmap(trust_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('信任度量表各维度相关性分析', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ 信任度相关性热图已保存到: {save_path}")

def generate_trust_report(trust_stats, trust_dimensions, save_path='results/trust_scale_report.txt'):
    """生成信任度分析文字报告"""
    print("正在生成信任度分析报告...")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("自动化信任度量表（Trust in Automation Scale）分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 总体信任度
        overall_data = trust_stats[trust_stats['分析维度'] == '总体']
        if not overall_data.empty:
            overall_score = overall_data['平均信任度'].iloc[0]
            overall_std = overall_data['标准差'].iloc[0]
            overall_desc = overall_data['信任水平描述'].iloc[0]
            
            f.write("1. 总体信任度分析\n")
            f.write("-" * 30 + "\n")
            f.write(f"• 平均信任度: {overall_score:.2f} (SD = {overall_std:.2f})\n")
            f.write(f"• 信任水平: {overall_desc}\n")
            f.write(f"• 样本数: {overall_data['样本数'].iloc[0]}人\n")
            f.write(f"• 分数范围: {overall_data['最小值'].iloc[0]:.2f} - {overall_data['最大值'].iloc[0]:.2f}\n\n")
        
        # 按科室分析
        f.write("2. 按科室信任度分析\n")
        f.write("-" * 30 + "\n")
        dept_data = trust_stats[trust_stats['分析维度'].str.contains('科室_')]
        for _, row in dept_data.iterrows():
            dept_name = row['分组']
            f.write(f"• {dept_name}:\n")
            f.write(f"  - 平均信任度: {row['平均信任度']:.2f} ({row['信任水平描述']})\n")
            f.write(f"  - 样本数: {row['样本数']}人\n")
            f.write(f"  - 标准差: {row['标准差']:.2f}\n\n")
        
        # 按年资分析
        f.write("3. 按年资信任度分析\n")
        f.write("-" * 30 + "\n")
        seniority_data = trust_stats[trust_stats['分析维度'].str.contains('年资_')]
        for _, row in seniority_data.iterrows():
            f.write(f"• {row['分组']}:\n")
            f.write(f"  - 平均信任度: {row['平均信任度']:.2f} ({row['信任水平描述']})\n")
            f.write(f"  - 样本数: {row['样本数']}人\n")
            f.write(f"  - 标准差: {row['标准差']:.2f}\n\n")
        
        # AI使用顺序分析
        f.write("4. AI使用顺序对信任度的影响\n")
        f.write("-" * 30 + "\n")
        ai_data = trust_stats[trust_stats['分析维度'] == 'AI使用顺序']
        for _, row in ai_data.iterrows():
            f.write(f"• {row['分组']}:\n")
            f.write(f"  - 平均信任度: {row['平均信任度']:.2f} ({row['信任水平描述']})\n")
            f.write(f"  - 样本数: {row['样本数']}人\n")
        
        if not ai_data.empty and not pd.isna(ai_data['p_value'].iloc[0]):
            t_val = ai_data['t_value'].iloc[0]
            p_val = ai_data['p_value'].iloc[0]
            effect_size = ai_data['effect_size'].iloc[0]
            f.write(f"• 统计检验结果:\n")
            f.write(f"  - t值: {t_val:.4f}\n")
            f.write(f"  - p值: {p_val:.4f}\n")
            f.write(f"  - 效应量 (Cohen's d): {effect_size:.4f}\n")
            if p_val < 0.05:
                f.write(f"  - 结论: AI使用顺序对信任度有显著影响\n\n")
            else:
                f.write(f"  - 结论: AI使用顺序对信任度无显著影响\n\n")
        
        # 维度分析
        f.write("5. 信任度各维度分析\n")
        f.write("-" * 30 + "\n")
        
        # 按平均分排序
        trust_dimensions_sorted = trust_dimensions.sort_values('平均分', ascending=False)
        
        f.write("信任度各维度得分排序（从高到低）:\n\n")
        for i, (_, row) in enumerate(trust_dimensions_sorted.iterrows(), 1):
            f.write(f"{i}. {row['信任维度']}\n")
            f.write(f"   平均分: {row['平均分']:.2f} (SD = {row['标准差']:.2f})\n")
            f.write(f"   分数范围: {row['最小值']:.0f} - {row['最大值']:.0f}\n\n")
        
        # 关键发现
        f.write("6. 关键发现与结论\n")
        f.write("-" * 30 + "\n")
        
        highest_dim = trust_dimensions_sorted.iloc[0]
        lowest_dim = trust_dimensions_sorted.iloc[-1]
        
        f.write(f"• 信任度最高的维度: {highest_dim['信任维度']} ({highest_dim['平均分']:.2f}分)\n")
        f.write(f"• 信任度最低的维度: {lowest_dim['信任维度']} ({lowest_dim['平均分']:.2f}分)\n")
        
        # 科室差异
        dept_data_sorted = dept_data.sort_values('平均信任度', ascending=False)
        if not dept_data_sorted.empty:
            highest_dept = dept_data_sorted.iloc[0]
            lowest_dept = dept_data_sorted.iloc[-1]
            f.write(f"• 信任度最高的科室: {highest_dept['分组']} ({highest_dept['平均信任度']:.2f}分)\n")
            f.write(f"• 信任度最低的科室: {lowest_dept['分组']} ({lowest_dept['平均信任度']:.2f}分)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 60 + "\n")
    
    print(f"✓ 信任度分析报告已保存到: {save_path}")

def main():
    """主函数"""
    print("=== 自动化信任度量表可视化分析 ===")
    
    # 设置绘图样式
    setup_plotting_style()
    
    # 加载数据
    trust_data, trust_stats, trust_dimensions = load_trust_data()
    
    # 生成可视化
    plot_trust_overview(trust_stats)
    plot_trust_dimensions(trust_dimensions)
    plot_trust_correlation_heatmap(trust_data)
    
    # 生成文字报告
    generate_trust_report(trust_stats, trust_dimensions)
    
    print("\n✅ 自动化信任度量表可视化分析完成！")
    print("生成的文件:")
    print("  - results/trust_scale_overview.png")
    print("  - results/trust_scale_dimensions.png") 
    print("  - results/trust_scale_correlation.png")
    print("  - results/trust_scale_report.txt")

if __name__ == "__main__":
    main()
