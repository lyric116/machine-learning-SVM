### 附录D：图表索引

本实验生成了以下可视化图表，用于辅助分析：

#### 数据分析图表

1. **01_raw_monthly_user_type.png**
   - 原始数据各月会员/散户数量分布
   - 展示12个月的时间趋势
   - 发现：会员数量总体稳定，散户在夏季增多

2. **02_clean_sample_distribution.png**
   - 清洗后分层样本在12个月中的分布
   - 验证抽样的均匀性
   - 每月约4,867条样本，分布均衡

3. **03_duration_by_user_type.png**
   - 骑行时长按用户类型的分布
   - 箱型图展示四分位数
   - 清晰显示散户时长更长

4. **04_distance_by_user_type.png**
   - 骑行距离按用户类型的分布
   - 发现90.1%重叠，是性能瓶颈

5. **05_hour_weekday_heatmap.png**
   - 小时×星期的骑行热力图
   - 展示通勤高峰模式
   - 工作日7-9点、17-19点明显

6. **06_start_location_scatter.png**
   - 起点空间散点图（彩色标记用户类型）
   - 展示空间聚类模式
   - 市中心会员密集，景点散户多

#### 模型评估图表

7. **07_confusion_matrix.png**
   - 测试集混淆矩阵热力图
   - 数值标注清晰
   - 对角线颜色深，分类效果可视化

8. **08_permutation_importance.png**
   - 置换特征重要性条形图
   - Top-20特征排序
   - start_grid最重要

9. **09_svm_cv_comparison.png**
   - 不同C参数下的交叉验证F1对比
   - 折线图展示趋势
   - C=0.02最优

#### 流程图

10. **10_experiment_workflow.png**
    - 实验整体流程图
    - 从数据抽样到结果分析
    - 清晰展示各步骤关系

11. **11_svm_workflow.png**
    - SVM算法流程图
    - 包含数学公式
    - 展示优化过程

所有图表均保存在`outputs/figures/`目录下。
