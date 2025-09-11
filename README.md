# AICare 临床决策支持系统 - CHI论文数据分析

## 项目概述

本仓库包含CHI论文的数据分析代码和文档：

**"Augmenting Clinical Decision-Making with an Interactive and Interpretable AI Copilot: A Real-World User Study with Physicians in Nephrology and Obstetrics"**

## 研究背景

我们在2家医院的3个科室对AICare系统进行了全面的用户研究：

1. **BY - 产科**: 预测孕妇自发性早产
2. **BY - 肾内科**: 预测ESRD患者1年内死亡风险
3. **XY - 肾内科**: 预测ESRD患者1年内死亡风险

## AICare系统功能

AICare系统提供4个主要的可解释AI功能来辅助临床决策：

1. **动态风险轨迹可视化**: 显示患者每次就诊时风险预测的折线图
2. **交互式个体化关键指标分析**: 每次就诊的特征重要性分析，支持交互式探索
3. **人群级别指标分析可视化**: 数据集中特征重要性、数值和患者风险的3D和2D可视化
4. **大语言模型驱动的诊疗建议**: 基于患者EHR数据、风险轨迹和临床指标，使用DeepSeek-V3.1提供分析和建议

## 研究设计

### 方法学
- **被试内设计**: 每位参与者都体验了两种条件（有AI辅助和无AI辅助）
- **平衡设计**: 参与者随机分配到不同顺序组以消除顺序效应
- **样本量**: 每个科室10个患者病例（测试集中的正负样本平衡）

### 实验条件分配规则
**案例时间数据条件分配**：
- 每位参与者分析10个患者案例，记录为 `case01_time` 到 `case10_time`
- 条件分配基于 `是否先使用AI分析系统` 变量：
  - 如果 `ai_first=True`（先用AI）：
    - `case01_time` 到 `case05_time`：AI辅助条件
    - `case06_time` 到 `case10_time`：无辅助条件
  - 如果 `ai_first=False`（后用AI）：
    - `case01_time` 到 `case05_time`：无辅助条件
    - `case06_time` 到 `case10_time`：AI辅助条件

**统计分析方法**：
- 由于采用被试内设计，所有效率、信心和认知负荷分析都使用**配对样本t检验** (`ttest_rel`)
- 每个参与者的两种条件数据形成配对，消除个体差异的影响

### 数据收集
1. **研究前**: 系统演示和背景问卷调查
2. **临床分析**: 
   - 条件A（对照组）: 无AI辅助的分析
   - 条件B（实验组）: 使用AICare系统的分析
   - 每个条件后进行NASA-TLX工作负荷评估
3. **研究后**: 系统易用性量表(SUS)、自动化信任量表、AICare功能反馈
4. **可选**: 半结构化访谈（特别是高年资医生）

## 仓库结构

```
xiaoya-data-analysis/
├── data/                          # 原始数据文件（因隐私保护已git忽略）
│   ├── 问卷收集情况.xlsx             # 人工记录的问卷实施情况
│   └── 问卷数据.xlsx                # 收集到的问卷回答数据
├── scripts/                       # 数据处理和分析脚本
│   ├── simple_merge.py             # 简单数据合并脚本
│   ├── inspect_excel_headers.py    # Excel表头检查工具
│   └── extract_headers.py          # 表头提取和持久化工具
├── results/                       # 分析输出和可视化结果
│   ├── merged_dataset_simple.xlsx  # 合并后的问卷数据
│   ├── dataset_headers.json        # 基础表头信息
│   └── dataset_headers_categorized.json # 分类表头信息
├── docs/                         # 附加文档
└── README.md                     # 本文件
```

## 数据文件详细说明

### 原始数据文件（data/ 目录）

#### `问卷收集情况.xlsx`
- **文件说明**: 人工记录的问卷实施元数据
- **数据结构**: 8行 × 多列
- **主要内容**:
  - 参与者ID和基本信息
  - 问卷开始和结束时间
  - 访谈记录和备注
  - 实验条件和随机化信息
- **用途**: 提供问卷收集过程的上下文信息，用于质量控制和数据验证

#### `问卷数据.xlsx`
- **文件说明**: 实际收集到的问卷回答数据
- **数据结构**: 16行（参与者）× 74列（问卷问题）
- **主要内容**:
  - 参与者基本信息（性别、年龄、职称、工作年限等）
  - 10个患者的风险评估结果和信心度评分
  - NASA-TLX工作负荷量表评分
  - 系统易用性量表(SUS)评分
  - 自动化信任量表评分
  - AICare系统功能反馈评分
- **用途**: 核心分析数据，包含所有研究变量和测量指标

### 处理后数据文件（results/ 目录）

#### `merged_dataset_simple.xlsx`
- **文件说明**: 合并并整理后的完整数据集
- **生成方式**: 通过`simple_merge.py`脚本按ID列合并两个原始文件
- **数据结构**: 16行（参与者）× 120列（合并后的所有变量）
- **主要改进**:
  - 按ID精确匹配合并数据
  - 统一的列命名规范
  - 合理的列顺序排列
  - 重复列自动处理（添加后缀）
- **用途**: 后续所有统计分析和可视化的基础数据文件

#### `dataset_headers.json`
- **文件说明**: 详细的表头信息和数据类型描述
- **生成方式**: 通过`extract_headers.py`脚本自动提取
- **数据结构**: JSON格式，包含每列的详细元信息
- **主要内容**:
  - 文件元数据（生成时间、行列数等）
  - 每列的详细信息：
    - 列索引和名称
    - 数据类型
    - 非空值和空值数量
    - 唯一值数量
    - 数值列的统计信息（最小值、最大值、均值）
    - 分类列的唯一值列表（≤20个唯一值时）
- **用途**: 
  - 快速了解数据结构
  - 编程时的数据类型参考
  - 数据质量检查
  - 自动化分析脚本的输入

#### `dataset_headers_categorized.json`
- **文件说明**: 按功能分类整理的表头信息
- **生成方式**: 通过`extract_headers.py`脚本基于列名关键词自动分类
- **数据结构**: JSON格式，按研究变量类型分组
- **分类体系**:
  - **participant_info** (5列): 参与者基本信息
    - ID、性别、年龄段、职称、工作年限等
  - **risk_assessment** (52列): 风险评估相关
    - 10个患者的风险评估、信心度、重要指标选择等
  - **nasa_tlx** (12列): NASA-TLX工作负荷量表
    - 脑力需求、体力需求、时间压力、任务表现、努力程度、挫败感
  - **sus_scale** (1列): 系统易用性量表汇总
  - **trust_scale** (1列): 自动化信任量表汇总
  - **system_feedback** (3列): 系统功能反馈
    - 动态风险轨迹、个体化指标分析、人群级分析、LLM建议
  - **collection_metadata** (3列): 收集情况元数据
    - 时间记录、访谈记录等
  - **other** (43列): 其他未分类列
    - 具体的SUS和信任量表题目、时间记录等
- **用途**:
  - 按研究维度快速筛选变量
  - 分类统计分析
  - 生成分组报告
  - 可视化设计的变量选择

### 数据使用示例

```python
import pandas as pd
import json

# 读取合并后的主数据文件
df = pd.read_excel('results/merged_dataset_simple.xlsx')

# 读取分类表头信息
with open('results/dataset_headers_categorized.json', 'r', encoding='utf-8') as f:
    headers_info = json.load(f)

# 获取NASA-TLX相关列
nasa_tlx_cols = [col['name'] for col in headers_info['categories']['nasa_tlx']]
nasa_tlx_data = df[nasa_tlx_cols]

# 获取参与者基本信息
participant_cols = [col['name'] for col in headers_info['categories']['participant_info']]
participant_data = df[participant_cols]
```

## 环境配置和使用指南

### 环境要求
- Python 3.11+
- UV包管理器（推荐）

### 快速开始

#### 1. 环境初始化
```bash
# 克隆仓库
git clone <repository-url>
cd xiaoya-data-analysis

# 激活UV虚拟环境
source .venv/bin/activate

# 同步依赖（如果需要）
uv sync
```

#### 2. 数据处理流程
```bash
# 步骤1: 检查原始Excel文件结构
python scripts/inspect_excel_headers.py

# 步骤2: 合并问卷数据
python scripts/simple_merge.py

# 步骤3: 提取表头信息并分类
python scripts/extract_headers.py
```

#### 3. 脚本说明

##### `scripts/inspect_excel_headers.py`
- **功能**: 检查Excel文件的表头结构和数据概览
- **输出**: 控制台显示每个文件的详细信息
- **用途**: 数据探索和质量检查

##### `scripts/simple_merge.py`
- **功能**: 按ID列合并两个Excel文件
- **输入**: `data/问卷收集情况.xlsx` + `data/问卷数据.xlsx`
- **输出**: `results/merged_dataset_simple.xlsx`
- **特点**: 简单高效，左连接合并，自动处理重复列

##### `scripts/extract_headers.py`
- **功能**: 提取表头信息并持久化为JSON文件
- **输入**: `results/merged_dataset_simple.xlsx`
- **输出**: 
  - `results/dataset_headers.json` - 基础表头信息
  - `results/dataset_headers_categorized.json` - 分类表头信息
- **特点**: 自动数据类型检测、分类整理、统计信息生成

### 数据分析工作流

1. **数据准备阶段**
   - 运行数据合并脚本
   - 检查数据质量和完整性
   - 生成表头信息文件

2. **探索性数据分析**
   - 基于分类表头信息选择变量
   - 生成描述性统计
   - 创建初步可视化

3. **统计分析**
   - NASA-TLX工作负荷分析
   - 系统易用性评估
   - 信任度分析
   - 功能反馈分析

4. **结果可视化**
   - 生成图表和报告
   - 保存到results目录

### 注意事项

- **数据隐私**: 原始数据文件已通过.gitignore排除，不会提交到版本控制
- **文件编码**: 所有脚本使用UTF-8编码处理中文内容
- **依赖管理**: 使用UV管理Python依赖，确保环境一致性
- **错误处理**: 所有脚本包含详细的错误处理和日志输出

### 常见问题

**Q: 运行脚本时提示找不到文件**
A: 确保原始Excel文件放在`data/`目录下，文件名完全匹配

**Q: 中文显示乱码**
A: 确保终端支持UTF-8编码，或在脚本中指定编码参数

**Q: 依赖包安装失败**
A: 使用`uv sync`重新同步依赖，或检查网络连接

### 贡献指南

1. 所有代码注释使用中文
2. 新增脚本需要更新README文档
3. 遵循现有的文件命名规范
4. 提交前运行所有脚本确保无错误

