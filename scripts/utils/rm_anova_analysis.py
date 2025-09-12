#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用rm-ANOVA分析模块
使用R的ARTool和afex包进行重复测量方差分析
替代原有的t检验分析

作者: AICare研究团队
日期: 2025年9月
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    R_AVAILABLE = True
    print("✓ rpy2可用，将使用R进行rm-ANOVA分析")
except ImportError:
    R_AVAILABLE = False
    print("❌ rpy2不可用！请安装rpy2和R环境")
    raise ImportError("rpy2模块未安装。请运行: pip install rpy2")

def setup_r_environment():
    """设置R环境和必需的包"""
    if not R_AVAILABLE:
        raise RuntimeError("R环境不可用，无法进行rm-ANOVA分析")
    
    try:
        ro.r('''
        # 设置语言环境
        Sys.setenv(LANGUAGE="en")
        Sys.setenv(LANG="en_US.UTF-8")
        
        # 加载必需的包
        suppressPackageStartupMessages({
            library(dplyr)
            library(tidyr)
            library(ARTool)    # ART RM-ANOVA
            library(emmeans)   # 事后比较
            library(afex)      # parametric RM-ANOVA (pes/ges)
            library(effectsize) # r_b CI（用于 Wilcoxon）
        })
        ''')
        print("✓ R环境和包加载成功")
        return True
    except Exception as e:
        print(f"❌ R包加载失败: {e}")
        print("请确保安装了所需的R包:")
        print("install.packages(c('dplyr', 'tidyr', 'ARTool', 'emmeans', 'afex', 'effectsize'))")
        raise RuntimeError(f"R包加载失败: {e}")

def perform_rm_anova_analysis(data_df, participant_col='participant_id', 
                             condition_col='condition', dv_col='accuracy',
                             grouping_vars=None):
    """
    执行重复测量方差分析
    
    参数:
    - data_df: 包含配对数据的DataFrame
    - participant_col: 参与者ID列名
    - condition_col: 条件列名（如'AI辅助', '无辅助'）
    - dv_col: 因变量列名（如'accuracy'）
    - grouping_vars: 分组变量列表（如['seniority']）
    
    返回:
    - results: 包含主效应和事后比较结果的字典
    """
    
    if not R_AVAILABLE:
        raise RuntimeError("R环境不可用，无法进行rm-ANOVA分析")
    
    # 设置R环境
    setup_r_environment()
    
    try:
        # 准备数据
        analysis_data = data_df[[participant_col, condition_col, dv_col]].copy()
        if grouping_vars:
            for var in grouping_vars:
                if var in data_df.columns:
                    analysis_data[var] = data_df[var]
        
        # 移除缺失值
        analysis_data = analysis_data.dropna()
        
        if len(analysis_data) == 0:
            print("⚠️ 没有有效数据进行分析")
            return None
        
        # 传入R
        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["analysis_data"] = ro.conversion.py2rpy(analysis_data)
        
        # 构建R分析脚本
        r_script = f'''
        # 数据预处理
        analysis_data${participant_col} <- as.factor(analysis_data${participant_col})
        analysis_data${condition_col} <- as.factor(analysis_data${condition_col})
        analysis_data${dv_col} <- as.numeric(analysis_data${dv_col})
        
        # 检查数据
        cat("数据维度:", dim(analysis_data), "\\n")
        cat("条件水平:", levels(analysis_data${condition_col}), "\\n")
        cat("参与者数:", length(unique(analysis_data${participant_col})), "\\n")
        
        # 聚合数据（被试×条件）
        agg_data <- analysis_data %>%
            dplyr::group_by({participant_col}, {condition_col}) %>%
            dplyr::summarise({dv_col}_agg = mean({dv_col}, na.rm = TRUE), .groups = "drop")
        
        # ART RM-ANOVA（主检验）
        tryCatch({{
            m_art <- art({dv_col}_agg ~ {condition_col} + Error({participant_col}/{condition_col}), data = agg_data)
            res_art <- as.data.frame(anova(m_art))
            
            # ART 事后比较
            mA <- artlm(m_art, "{condition_col}")
            emm <- emmeans(mA, ~ {condition_col})
            post_art_holm <- as.data.frame(pairs(emm, adjust = "holm"))
            post_art_bh <- as.data.frame(pairs(emm, adjust = "BH"))
            
            # 分割对比名称
            split_contrast <- function(df_) {{
                if (!("contrast" %in% names(df_))) return(df_)
                parts <- strsplit(as.character(df_$contrast), " - ")
                df_$group1 <- vapply(parts, `[`, "", 1)
                df_$group2 <- vapply(parts, `[`, "", 2)
                df_
            }}
            post_art_holm <- split_contrast(post_art_holm)
            post_art_bh <- split_contrast(post_art_bh)
            
            # Parametric RM-ANOVA（用于效应量）
            aov_pes <- afex::aov_ez(
                id = "{participant_col}",
                dv = "{dv_col}_agg",
                within = "{condition_col}",
                data = agg_data,
                type = 3,
                anova_table = list(correction = "none", es = "pes")
            )
            res_rm_pes <- as.data.frame(aov_pes$anova_table)
            
            aov_ges <- afex::aov_ez(
                id = "{participant_col}",
                dv = "{dv_col}_agg", 
                within = "{condition_col}",
                data = agg_data,
                type = 3,
                anova_table = list(correction = "none", es = "ges")
            )
            res_rm_ges <- as.data.frame(aov_ges$anova_table)
            
            # 描述性统计
            desc_stats <- agg_data %>%
                dplyr::group_by({condition_col}) %>%
                dplyr::summarise(
                    mean_val = mean({dv_col}_agg, na.rm = TRUE),
                    sd_val = sd({dv_col}_agg, na.rm = TRUE),
                    n_subj = dplyr::n(),
                    .groups = "drop"
                ) %>%
                dplyr::rename(condition = {condition_col})
            
            art_success <- TRUE
            
        }}, error = function(e) {{
            cat("ART分析失败:", e$message, "\\n")
            art_success <<- FALSE
            res_art <<- data.frame()
            post_art_holm <<- data.frame()
            post_art_bh <<- data.frame()
            res_rm_pes <<- data.frame()
            res_rm_ges <<- data.frame()
            desc_stats <<- data.frame()
        }})
        '''
        
        # 执行R分析
        ro.r(r_script)
        
        # 获取结果
        with localconverter(ro.default_converter + pandas2ri.converter):
            art_success = ro.r('art_success')[0]
            
            if art_success:
                results = {
                    'main_effect': ro.conversion.rpy2py(ro.r('res_art')),
                    'post_hoc_holm': ro.conversion.rpy2py(ro.r('post_art_holm')),
                    'post_hoc_bh': ro.conversion.rpy2py(ro.r('post_art_bh')),
                    'effect_size_pes': ro.conversion.rpy2py(ro.r('res_rm_pes')),
                    'effect_size_ges': ro.conversion.rpy2py(ro.r('res_rm_ges')),
                    'descriptive_stats': ro.conversion.rpy2py(ro.r('desc_stats')),
                    'analysis_type': 'ART_RM_ANOVA'
                }
                
                print("✓ ART RM-ANOVA分析完成")
                return results
            else:
                print("❌ ART分析失败")
                raise RuntimeError("ART RM-ANOVA分析失败，请检查数据和R环境")
                
    except Exception as e:
        print(f"❌ R分析出错: {e}")
        raise RuntimeError(f"rm-ANOVA分析失败: {e}")

# t检验回退方案已移除，现在要求必须使用rm-ANOVA

def format_rm_anova_results(results):
    """
    格式化rm-ANOVA结果用于报告
    """
    if not results:
        return "分析失败"
    
    report = []
    
    # 主效应
    if 'main_effect' in results and len(results['main_effect']) > 0:
        main = results['main_effect']
        if 'F' in main.columns and 'Pr(>F)' in main.columns:
            f_val = main['F'].iloc[0] if len(main) > 0 else np.nan
            p_val = main['Pr(>F)'].iloc[0] if len(main) > 0 else np.nan
            
            report.append(f"主效应: F = {f_val:.3f}, p = {p_val:.4f}")
            
            if p_val < 0.001:
                sig_text = "p < 0.001"
            elif p_val < 0.01:
                sig_text = "p < 0.01"
            elif p_val < 0.05:
                sig_text = "p < 0.05"
            else:
                sig_text = "n.s."
            
            report.append(f"显著性: {sig_text}")
    
    # 效应量
    if 'effect_size_pes' in results and len(results['effect_size_pes']) > 0:
        pes = results['effect_size_pes']
        if 'pes' in pes.columns:
            pes_val = pes['pes'].iloc[0] if len(pes) > 0 else np.nan
            report.append(f"偏η² = {pes_val:.3f}")
    
    # 事后比较
    if 'post_hoc_holm' in results and len(results['post_hoc_holm']) > 0:
        post = results['post_hoc_holm']
        report.append("\n事后比较 (Holm校正):")
        for idx, row in post.iterrows():
            if 'group1' in row and 'group2' in row and 'p.value' in row:
                p_val = row['p.value']
                estimate = row.get('estimate', np.nan)
                report.append(f"  {row['group1']} vs {row['group2']}: 差异 = {estimate:.4f}, p = {p_val:.4f}")
    
    # 分析类型
    analysis_type = results.get('analysis_type', 'UNKNOWN')
    report.append(f"\n分析方法: {analysis_type}")
    
    return "\n".join(report)

def print_rm_anova_summary(results, title="RM-ANOVA分析结果"):
    """
    打印rm-ANOVA结果摘要
    """
    print(f"\n=== {title} ===")
    
    if not results:
        print("❌ 分析失败或无有效数据")
        return
    
    # 描述性统计
    if 'descriptive_stats' in results and len(results['descriptive_stats']) > 0:
        print("\n📊 描述性统计:")
        desc = results['descriptive_stats']
        print(f"DEBUG: 描述性统计数据类型: {type(desc)}")
        print(f"DEBUG: 列名: {desc.columns.tolist() if hasattr(desc, 'columns') else '无'}")
        
        for idx, row in desc.iterrows():
            print(f"DEBUG: 处理第{idx}行: {dict(row)}")
            condition = row.get('condition', f'条件{idx}')
            mean_val = row.get('mean_val', np.nan)
            sd_val = row.get('sd_val', np.nan)
            n_subj = row.get('n_subj', 0)
            
            print(f"DEBUG: 数据类型 - condition: {type(condition)}, mean: {type(mean_val)}, sd: {type(sd_val)}, n: {type(n_subj)}")
            
            try:
                print(f"  {condition}: M = {float(mean_val):.4f}, SD = {float(sd_val):.4f}, n = {int(n_subj)}")
            except (ValueError, TypeError) as e:
                print(f"DEBUG: 格式化错误: {e}")
                print(f"  {condition}: M = {mean_val}, SD = {sd_val}, n = {n_subj}")
    
    # 主效应结果
    if 'main_effect' in results and len(results['main_effect']) > 0:
        print("\n🔍 主效应:")
        main = results['main_effect']
        for idx, row in main.iterrows():
            term = row.get('Term', f'因子{idx}')
            f_val = row.get('F', np.nan)
            p_val = row.get('Pr(>F)', np.nan)
            
            sig_marker = ""
            if not pd.isna(p_val):
                if p_val < 0.001:
                    sig_marker = "***"
                elif p_val < 0.01:
                    sig_marker = "**"
                elif p_val < 0.05:
                    sig_marker = "*"
            
            print(f"  {term}: F = {f_val:.3f}, p = {p_val:.4f} {sig_marker}")
    
    # 效应量
    if 'effect_size_pes' in results and len(results['effect_size_pes']) > 0:
        pes_data = results['effect_size_pes']
        if 'pes' in pes_data.columns:
            pes_val = pes_data['pes'].iloc[0]
            print(f"  偏η² = {pes_val:.3f}")
    
    # 事后比较
    if 'post_hoc_holm' in results and len(results['post_hoc_holm']) > 0:
        print("\n📋 事后比较 (Holm校正):")
        post = results['post_hoc_holm']
        for idx, row in post.iterrows():
            if 'group1' in row and 'group2' in row:
                group1 = row['group1']
                group2 = row['group2'] 
                p_val = row.get('p.value', np.nan)
                estimate = row.get('estimate', np.nan)
                
                sig_marker = ""
                if not pd.isna(p_val):
                    if p_val < 0.001:
                        sig_marker = "***"
                    elif p_val < 0.01:
                        sig_marker = "**"
                    elif p_val < 0.05:
                        sig_marker = "*"
                
                print(f"  {group1} vs {group2}: 差异 = {estimate:.4f}, p = {p_val:.4f} {sig_marker}")
    
    # 分析类型
    analysis_type = results.get('analysis_type', 'UNKNOWN')
    print(f"\n🔧 分析方法: {analysis_type}")
    
    if analysis_type == 'PAIRED_T_TEST':
        cohens_d = results.get('cohens_d', np.nan)
        print(f"   Cohen's d = {cohens_d:.3f}")

# 测试函数
def test_rm_anova():
    """测试rm-ANOVA分析功能"""
    print("测试rm-ANOVA分析模块...")
    
    try:
        # 检查R环境
        if not R_AVAILABLE:
            raise ImportError("rpy2模块未安装")
        
        # 检查R包
        setup_r_environment()
        
        # 创建测试数据
        np.random.seed(42)
        n_participants = 10
        
        test_data = []
        for i in range(n_participants):
            # AI辅助条件
            ai_score = np.random.normal(0.7, 0.15)  # 平均70%准确率
            test_data.append({
                'participant_id': f'P{i+1:02d}',
                'condition': 'AI辅助',
                'accuracy': max(0, min(1, ai_score)),
                'seniority': '高年资' if i < 5 else '低年资'
            })
            
            # 无辅助条件
            no_ai_score = np.random.normal(0.6, 0.15)  # 平均60%准确率
            test_data.append({
                'participant_id': f'P{i+1:02d}',
                'condition': '无辅助',
                'accuracy': max(0, min(1, no_ai_score)),
                'seniority': '高年资' if i < 5 else '低年资'
            })
        
        test_df = pd.DataFrame(test_data)
        
        # 执行分析
        results = perform_rm_anova_analysis(
            test_df,
            participant_col='participant_id',
            condition_col='condition',
            dv_col='accuracy'
        )
        
        # 显示结果
        print_rm_anova_summary(results, "测试分析结果")
        
        return results
        
    except ImportError as e:
        print(f"❌ 环境检查失败: {e}")
        print("\n🔧 解决方案:")
        print("1. 安装系统依赖: apt install -y libtirpc-dev libpcre2-dev libbz2-dev liblzma-dev libicu-dev r-base-dev")
        print("2. 安装rpy2: pip install rpy2")
        print("3. 安装R包: install.packages(c('dplyr', 'tidyr', 'ARTool', 'emmeans', 'afex', 'effectsize'))")
        raise
    except RuntimeError as e:
        print(f"❌ 分析失败: {e}")
        print("\n🔧 检查事项:")
        print("1. R环境是否正确安装")
        print("2. 所需R包是否已安装")
        print("3. 数据格式是否正确")
        raise
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        raise

if __name__ == "__main__":
    test_rm_anova()
