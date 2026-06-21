### 附录F：实验环境与软件配置

#### F.1 硬件环境

- **操作系统**：Windows WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- **发行版**：Ubuntu 22.04 LTS
- **CPU**：多核处理器（支持并行计算）
- **内存**：充足内存用于大规模数据处理
- **存储**：SSD存储，加快数据读写

#### F.2 软件环境

**Python环境**：
- Python版本：3.14
- 包管理：conda (miniconda3)
- 虚拟环境：cvlab

**核心依赖库**：
```
pandas>=2.0.0          # 数据处理
numpy>=1.24.0          # 数值计算
scikit-learn>=1.3.0    # 机器学习算法
matplotlib>=3.7.0      # 数据可视化
seaborn>=0.12.0        # 统计图表
joblib>=1.3.0          # 模型序列化
```

**安装命令**：
```bash
conda create -n cvlab python=3.14
conda activate cvlab
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

#### F.3 项目代码结构

本实验代码组织清晰，模块化设计：

```
machine_learning/
├── data/                           # 原始数据目录
│   ├── 202501/                    # 1月数据
│   ├── 202502/                    # 2月数据
│   └── ...                        # 其他月份
│
├── scripts/                        # 核心代码
│   ├── 01_data_sampling.py        # 数据抽样
│   ├── 02_data_cleaning.py        # 数据清洗
│   ├── 03_feature_engineering.py  # 特征工程
│   ├── 04_train_model.py          # 模型训练（主程序）
│   ├── 05_evaluate_model.py       # 模型评估
│   ├── 06_feature_importance.py   # 特征重要性
│   └── 07_visualization.py        # 可视化
│
├── outputs/                        # 输出结果
│   ├── processed/                 # 处理后的数据
│   │   └── cleaned_stratified_sample.csv
│   ├── models/                    # 训练好的模型
│   │   ├── svm_member_casual_pipeline.joblib
│   │   └── scaler.joblib
│   ├── figures/                   # 图表
│   │   ├── 01_raw_monthly_user_type.png
│   │   ├── 07_confusion_matrix.png
│   │   └── ...
│   ├── tables/                    # 结果表格
│   │   ├── classification_report.csv
│   │   ├── confusion_matrix.csv
│   │   └── svm_cv_results.csv
│   └── metrics.json              # 性能指标
│
├── final_submission/              # 最终提交
│   ├── report/                   # 实验报告
│   │   └── 完整实验报告.md
│   ├── code/                     # 整理后的代码
│   └── figures/                  # 关键图表
│
├── environment.yml               # conda环境配置
└── README.md                     # 项目说明
```

**代码设计原则**：
1. **模块化**：每个脚本负责一个具体任务
2. **可复现**：固定随机种子，记录参数
3. **可读性**：详细注释，清晰命名
4. **可扩展**：便于添加新特征、新模型
