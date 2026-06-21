## 参考文献

[1] Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.

[2] Chang, C. C., & Lin, C. J. (2011). LIBSVM: a library for support vector machines. ACM transactions on intelligent systems and technology (TIST), 2(3), 1-27.

[3] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.

[4] Capital Bikeshare. (2025). System Data. Retrieved from https://www.capitalbikeshare.com/system-data

[5] Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

[6] Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. ACM SIGKDD Explorations Newsletter, 3(1), 27-32.

[7] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.

---

## 附录

### 附录A：数据字段说明

| 字段名 | 数据类型 | 取值范围 | 缺失率 | 说明 |
|--------|----------|----------|--------|------|
| ride_id | string | - | 0% | 骑行唯一标识 |
| rideable_type | string | classic/electric/docked | 0% | 车辆类型 |
| started_at | datetime | 2025-01-01 to 2025-12-31 | 0% | 开始时间 |
| ended_at | datetime | 2025-01-01 to 2025-12-31 | 0% | 结束时间 |
| start_station_name | string | - | 12.3% | 起点站点名称 |
| start_station_id | string | - | 12.3% | 起点站点ID |
| end_station_name | string | - | 14.7% | 终点站点名称 |
| end_station_id | string | - | 14.7% | 终点站点ID |
| start_lat | float | 38.79-39.0 | 0.1% | 起点纬度 |
| start_lng | float | -77.12--76.91 | 0.1% | 起点经度 |
| end_lat | float | 38.79-39.0 | 0.1% | 终点纬度 |
| end_lng | float | -77.12--76.91 | 0.1% | 终点经度 |
| member_casual | string | member/casual | 0% | 用户类型（目标） |

### 附录B：超参数搜索结果

**GridSearchCV完整结果**（部分）：

| C | loss | class_weight | CV F1 | 排名 |
|---|------|--------------|-------|------|
| 0.02 | squared_hinge | balanced | 0.6381 | 1 |
| 0.03 | squared_hinge | balanced | 0.6378 | 2 |
| 0.05 | squared_hinge | balanced | 0.6371 | 3 |
| 0.02 | hinge | balanced | 0.6365 | 4 |
| 0.1 | squared_hinge | balanced | 0.6359 | 5 |

完整结果共36种组合，最佳为C=0.02。

### 附录C：混淆矩阵详细分析

**测试集混淆矩阵（11,681个样本）**：

```
                预测Casual  预测Member   合计
实际Casual         3244        2565      5809
实际Member         1624        4248      5872
合计               4868        6813     11681
```

**各类别指标**：

**Casual类**：
- True Positives (TP): 3,244
- False Positives (FP): 1,624
- False Negatives (FN): 2,565
- True Negatives (TN): 4,248
- Precision: 3244/(3244+1624) = 0.67
- Recall: 3244/(3244+2565) = 0.56
- F1: 2×0.67×0.56/(0.67+0.56) = 0.61

**Member类**：
- True Positives (TP): 4,248
- False Positives (FP): 2,565
- False Negatives (FN): 1,624
- True Negatives (TN): 3,244
- Precision: 4248/(4248+2565) = 0.62
- Recall: 4248/(4248+1624) = 0.72
- F1: 2×0.62×0.72/(0.62+0.72) = 0.67
