# R环境配置指南

本指南帮助您配置R环境以支持rm-ANOVA分析功能。

## 快速开始

如果您不想配置R环境，**系统会自动回退到t检验**，所有功能仍可正常使用。

## 完整安装（推荐）

### 1. 安装R

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install r-base r-base-dev
```

#### CentOS/RHEL
```bash
sudo yum install R
```

#### Windows
下载并安装：https://cran.r-project.org/bin/windows/base/

#### macOS
```bash
brew install r
```

### 2. 安装Python rpy2包
```bash
pip install rpy2
```

### 3. 安装必需的R包

启动R并运行：
```r
# 安装必需的包
install.packages(c("dplyr", "tidyr", "emmeans", "effectsize"))

# 安装ARTool包（用于ART RM-ANOVA）
install.packages("ARTool")

# 安装afex包（用于参数化RM-ANOVA和效应量）
install.packages("afex")
```

### 4. 验证安装

运行测试脚本：
```bash
python test_rm_anova.py
```

如果看到"ART RM-ANOVA分析完成"，说明安装成功！

## 故障排除

### 问题1: rpy2导入失败
```
ImportError: No module named 'rpy2'
```
**解决方案**: `pip install rpy2`

### 问题2: R包缺失
```
Error: package 'ARTool' is not installed
```
**解决方案**: 在R中运行 `install.packages("ARTool")`

### 问题3: R版本过旧
```
Error: R version too old
```
**解决方案**: 更新R到最新版本

### 问题4: 权限问题
```
Permission denied when installing packages
```
**解决方案**: 
- Linux/Mac: 使用sudo或配置个人包库
- Windows: 以管理员身份运行

## 系统兼容性

- ✅ **无R环境**: 自动回退到t检验，功能完整
- ✅ **有R环境**: 使用高级rm-ANOVA分析
- ✅ **WSL**: 完全支持
- ✅ **Docker**: 支持（需要安装R）

## 分析方法对比

| 环境状态 | 使用方法 | 效应量 | 事后比较 |
|---------|---------|--------|----------|
| 无R | 配对t检验 | Cohen's d | - |
| 有R | ART RM-ANOVA | 偏η² | Holm/BH校正 |

## 性能优势

使用rm-ANOVA相比t检验的优势：
1. **更严格的统计假设检验**
2. **更准确的效应量估计**
3. **标准化的事后比较**
4. **更好的多重比较控制**
5. **符合重复测量设计的统计要求**

## 技术说明

- **ART RM-ANOVA**: 对非正态数据进行Aligned Rank Transform
- **参数化RM-ANOVA**: 计算精确的效应量
- **智能回退**: 确保在任何环境下都能运行
- **数据转换**: 自动处理宽格式↔长格式转换

---

**需要帮助？** 
- 查看测试输出了解当前状态
- 所有分析脚本都有详细的错误提示
- 回退机制确保功能始终可用
