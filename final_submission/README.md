# Capital Bikeshare 用户分类实验

基于支持向量机(SVM)的Capital Bikeshare用户类型(会员/散户)分类实验。

## 📋 实验概述

- **任务**：根据骑行特征预测用户类型
- **算法**：支持向量机（SVM）
- **数据**：Capital Bikeshare 2025年1-12月数据
- **性能**：准确率 64.13%，F1分数 64.11%

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建conda环境
conda create -n bikeshare python=3.14
conda activate bikeshare

# 安装依赖
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### 2. 运行主程序

```bash
# 进入项目目录
cd machine_learning

# 运行主程序
python final_submission/code/main.py
```

### 3. 查看结果

运行完成后，结果保存在：
- `final_submission/models/svm_model.joblib` - 训练好的模型
- `final_submission/models/results.json` - 评估指标
- `final_submission/report/完整实验报告.md` - 详细报告

## 📁 项目结构

```
machine_learning/
├── data/                           # 原始数据（12个月CSV文件）
├── outputs/
│   ├── processed/                 # 处理后的数据
│   │   └── cleaned_stratified_sample.csv
│   ├── figures/                   # 可视化图表
│   ├── tables/                    # 结果表格
│   └── models/                    # 训练的模型
│
├── final_submission/              # 最终提交文件
│   ├── code/
│   │   └── main.py               # 主程序★
│   ├── models/                    # 输出模型
│   └── report/
│       └── 完整实验报告.md        # 实验报告★
│
└── README.md                      # 本文件
```

## 🔬 实验流程

1. **数据抽样**：从666万条记录中按月按类别分层抽样
2. **数据清洗**：过滤异常值，处理缺失值
3. **特征工程**：构造37个特征（时间、空间、行为、频率统计）
4. **数据划分**：80%训练集，20%测试集（分层划分）
5. **特征标准化**：Z-score标准化
6. **模型训练**：GridSearchCV + 3折交叉验证
7. **模型评估**：准确率、F1、混淆矩阵

## 📊 核心成果

| 指标 | 数值 |
|------|------|
| 测试集准确率 | 64.13% |
| 测试集F1分数 | 64.11% |
| 交叉验证F1 | 63.81% |
| 训练样本数 | 46,724 |
| 测试样本数 | 11,681 |
| 使用特征数 | 37 |

## 🎯 关键发现

1. **频率统计特征**：最大创新，贡献6.4%性能提升
2. **最小特征集**：仅用8个特征即可达到97.8%性能
3. **线性核最优**：LinearSVC优于RBF和多项式核
4. **空间因素最强**：起终点位置是最强预测因子

## 📖 详细报告

完整的实验报告（4000字）包含：
- 实验问题与目标
- 数据介绍与预处理
- SVM算法原理与流程
- 实验结果与分析
- 因素影响分析
- 有趣的发现

查看：`final_submission/report/完整实验报告.md`

## 🛠️ 主要技术

- **语言**：Python 3.14
- **核心库**：scikit-learn, pandas, numpy
- **算法**：LinearSVC (线性支持向量机)
- **优化**：GridSearchCV + StratifiedKFold
- **评估**：Accuracy, F1-Score, Confusion Matrix

## 📧 联系方式

- 作者：实验报告作者
- 日期：2026年6月

## 📄 许可证

本项目仅用于学术研究和课程实验。
