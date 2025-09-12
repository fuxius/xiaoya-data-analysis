# Results Directory Structure

本目录包含AICare研究项目的所有分析结果文件，按照分析类型进行组织。

## 目录结构说明

### 📊 核心分析结果

#### `/trust/` - 自动化信任度量表分析
- `trust_scale_data.csv` - 信任度原始数据（16名参与者的12个信任维度得分和总分）
- `trust_scale_analysis.csv` - 信任度统计分析结果（按总体、科室、年资、AI使用顺序的统计对比，含t检验结果）
- `trust_scale_dimensions.csv` - 信任度各维度分析（12个信任维度的描述统计和排序）

#### `/workload/` - NASA-TLX认知负荷分析
- `workload_raw_data.csv` - NASA-TLX原始数据（宽格式，每个参与者的两次测量条件）
- `workload_long_data.csv` - NASA-TLX长格式数据（AI辅助组vs无辅助组，每行一个条件）
- `workload_analysis.csv` - 认知负荷统计分析（总体、科室、年资分组的负荷对比，含配对t检验）
- `nasa_tlx_dimensions.csv` - NASA-TLX六个维度分析（脑力需求、体力需求、时间压力、任务表现、努力程度、挫败感）

#### `/confidence/` - 信心度分析
- `confidence_raw_data.csv` - 信心度原始数据（每个参与者对10个患者的信心评估，AI辅助vs无辅助）
- `confidence_long_data.csv` - 信心度长格式数据（每行一个评估条件）
- `confidence_analysis.csv` - 信心度统计分析（按总体、科室、年资、患者类型的信心对比，含配对t检验）
- `confidence_patterns.csv` - 信心模式分析（信心水平分布和变化模式）

#### `/efficiency/` - 效率分析
- `efficiency_raw_data.csv` - 效率原始数据（每个参与者对10个患者的分析时间，AI辅助vs无辅助）
- `efficiency_long_data.csv` - 效率长格式数据（每行一个分析任务的时间记录）
- `efficiency_analysis.csv` - 效率统计分析（按总体、科室、年资、患者类型的时间对比，含配对t检验）

#### `/sus/` - 系统易用性量表分析
- `sus_raw_data.csv` - SUS原始数据（16名参与者的10个SUS题目回答和计算得分）
- `sus_analysis.csv` - SUS统计分析（SUS总分和各题目的描述统计，含易用性等级评价）

#### `/performance/` - 临床医生性能分析
- `clinician_performance_raw_data.csv` - 临床医生表现原始数据（风险评估准确性）
- `clinician_performance_analysis_detailed.csv` - 详细性能分析（准确性、敏感性、特异性等指标）
- `clinician_binary_performance_raw.csv` - 二分类性能原始数据
- `clinician_binary_analysis_detailed.csv` - 二分类详细分析
- `clinician_auprc_raw_data.csv` - AUPRC性能原始数据
- `clinician_auprc_analysis_detailed.csv` - AUPRC详细分析
- `clinician_assessments_with_conditions.csv` - 带条件的评估数据（AI辅助vs无辅助）
- `ai_model_performance.csv` - AI模型基准性能
- `ai_model_binary_performance.csv` - AI模型二分类性能
- `ai_model_auprc_performance.csv` - AI模型AUPRC性能
- `performance_summary_and_insights.csv` - 性能总结和洞察
- `binary_performance_summary.csv` - 二分类性能总结
- `auprc_summary_and_insights.csv` - AUPRC性能总结
- `performance_insights.txt` - 性能分析洞察报告
- `binary_performance_insights.txt` - 二分类性能洞察
- `auprc_insights.txt` - AUPRC性能洞察

### 📁 基础数据

#### `/datasets/` - 数据集
- `merged_dataset_simple.xlsx` - 合并后的主数据集（16名参与者×120个变量）
- `merged_dataset.xlsx` - 原始合并数据集
- `dataset_headers.json` - 数据表头详细信息（每列的数据类型、统计信息、唯一值）
- `dataset_headers_categorized.json` - 按功能分类的表头信息

#### `/participants/` - 参与者信息
- `participant_basic_info.csv` - 参与者基本信息（ID、性别、年龄、职称、工作年限等）
- `participant_basic_info.xlsx` - 参与者基本信息Excel版本
- `participants_with_seniority.csv` - 包含年资分类的参与者信息
- `seniority_classification.json` - 年资分类规则和结果（高年资vs低年资）

#### `/mappings/` - 数据映射
- 数据编码和映射关系文件

### 📋 专项分析

#### `/features/` - 功能有用性分析
- `feature_usefulness_raw_data.csv` - 功能有用性原始数据（16名参与者对4个功能模块的评分）
- `feature_usefulness_analysis.csv` - 功能有用性统计分析（总分和各功能模块的描述统计）
- `feature_usefulness_mapping.json` - 功能评分映射规则

#### `/interviews/` - 半结构化访谈分析
- `interview_raw_data.csv` - 访谈原始数据（10名参与者的11个访谈问题回答）
- `interview_mapping.json` - 访谈问题映射和编码规则

#### `/rq2_performance_analysis/` - RQ2专项性能分析（研究问题2：AI辅助对临床医生表现的影响）
- `comprehensive_performance_comparison.csv` - 综合性能对比分析
- `paired_performance_data.csv` - 配对性能数据（AI辅助vs无辅助）
- `statistical_test_results.csv` - 统计检验结果汇总
- `clinician_group_statistics.csv` - 临床医生分组统计
- `clinician_individual_performance.csv` - 个体临床医生表现
- `ai_model_detailed_performance.csv` - AI模型详细性能指标
- `ai_model_optimized_performance.csv` - 优化后AI模型性能
- `ai_model_threshold_analysis.csv` - AI模型阈值分析
- `ai_model_threshold_summary.csv` - 阈值分析总结
- `threshold_details_all.csv` - 所有阈值详细信息
- `xy_nephrology_individual_performance.csv` - XY肾内科个体表现
- `xy_nephrology_paired_data.csv` - XY肾内科配对数据
- `xy_nephrology_paired_performance.csv` - XY肾内科配对性能
- `comprehensive_analysis_report.txt` - 综合分析报告
- `summary_report.txt` - 总结报告
- `ai_model_threshold_report.txt` - AI模型阈值报告
- `xy_nephrology_analysis_report.txt` - XY肾内科专项分析报告

## 文件命名规范

- **原始数据文件**: `*_data.csv`, `*_raw_data.csv`, `*_raw.csv` - 包含原始测量数据和计算得分
- **分析结果文件**: `*_analysis.csv`, `*_detailed.csv` - 统计分析结果，含描述统计和推断统计
- **长格式数据文件**: `*_long_data.csv` - 重塑后的长格式数据，便于统计分析
- **统计摘要文件**: `*_summary.csv`, `*_summary_and_insights.csv` - 关键结果摘要
- **洞察报告文件**: `*_insights.txt`, `*_report.txt` - 文字形式的分析洞察和结论
- **维度分析文件**: `*_dimensions.csv` - 多维度量表的各维度分析
- **模式分析文件**: `*_patterns.csv` - 数据模式和分布特征分析
- **映射文件**: `*_mapping.json` - 数据编码和分类映射规则
- **配对数据文件**: `*_paired_data.csv`, `*_paired_performance.csv` - 配对实验设计的数据

## 数据类型说明

### 📊 统计分析文件内容
- **描述统计**: 样本数、平均值、标准差、中位数、最小值、最大值
- **推断统计**: t检验结果（t值、p值）、效应量（Cohen's d）
- **分组对比**: 按科室、年资、AI使用顺序等维度的对比分析
- **信心区间**: 95%置信区间（部分文件包含）

### 🔍 原始数据文件内容
- **参与者信息**: ID、科室、年资分类、AI使用顺序
- **测量数据**: 量表得分、时间记录、评估结果
- **条件标记**: AI辅助组vs无辅助组、第一次vs第二次测量

### 📈 长格式数据文件内容
- **观测单位**: 每行代表一个测量条件（如一次风险评估、一次NASA-TLX测量）
- **条件变量**: condition_type（AI辅助组/无辅助组）
- **结果变量**: 具体的测量结果（分数、时间、准确性等）

## 使用说明

### 🚀 快速开始
1. **主数据集**: 使用 `datasets/merged_dataset_simple.xlsx` 作为所有分析的基础数据
2. **快速查看结果**: 查看各目录下的 `*_summary.csv` 和 `*_analysis.csv` 文件
3. **详细分析**: 查看 `*_detailed.csv` 和 `*_dimensions.csv` 文件
4. **洞察总结**: 阅读 `*_insights.txt` 和 `*_report.txt` 文件

### 📋 按研究问题使用
- **RQ1 (用户体验)**: 查看 `trust/`, `sus/`, `workload/`, `confidence/` 目录
- **RQ2 (性能影响)**: 查看 `performance/`, `rq2_performance_analysis/`, `efficiency/` 目录
- **RQ3 (功能价值)**: 查看 `features/`, `interviews/` 目录

### 🔧 数据分析工作流
1. **探索性分析**: 从 `*_raw_data.csv` 开始了解数据分布
2. **描述性统计**: 查看 `*_analysis.csv` 了解基本统计特征  
3. **推断统计**: 关注 `*_analysis.csv` 中的t检验和效应量
4. **深入分析**: 使用 `*_long_data.csv` 进行自定义统计分析
5. **结果解释**: 参考 `*_insights.txt` 和 `*_report.txt` 的专业解读

## 📈 数据统计概览

### 参与者信息
- **总参与者**: 16名临床医生
- **科室分布**: XY-Nephrology (8人), BC-Obstetrics (3人), BS-Nephrology (5人)
- **年资分布**: 高年资 (9人), 低年资 (7人)
- **AI使用顺序**: AI先用 (9人), AI后用 (7人)

### 主要测量指标
- **认知负荷**: NASA-TLX量表 (6个维度)
- **信任度**: Trust in Automation Scale (12个维度)
- **易用性**: SUS量表 (10个题目)
- **信心度**: 对10个患者的风险评估信心 (1-5分)
- **效率**: 对10个患者的分析时间 (秒)
- **功能有用性**: 4个系统功能模块评价
- **定性访谈**: 11个半结构化问题

### 关键发现摘要
- **认知负荷**: AI辅助组 (249.31) vs 无辅助组 (284.94), p=0.087
- **信任度**: 总体平均 5.18分 (中等信任水平)
- **易用性**: SUS平均分 63.91分 (可接受水平)
- **信心度**: AI辅助组显著提高 (3.71 vs 3.29, p=0.040)
- **效率**: AI辅助组用时更长但无显著差异 (155.6s vs 118.2s, p=0.202)

## 更新日期

最后更新: 2025年9月11日
