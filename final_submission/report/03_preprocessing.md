#### 3.2.2 数据标准化

**为什么需要标准化**：
- SVM算法对特征尺度敏感
- 不同特征的量纲和数值范围差异大（如：distance_km: 0-20，duration_min: 1-180）
- 标准化可以加速模型收敛

**标准化方法**：使用StandardScaler进行Z-score标准化

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

其中$\mu$是均值，$\sigma$是标准差。

**处理流程**：
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 训练集拟合并转换
X_test_scaled = scaler.transform(X_test)        # 测试集只转换
```

**注意事项**：
- 仅在训练集上fit，避免测试集信息泄漏
- 类别变量先One-Hot编码再标准化
- 目标变量不需要标准化

### 3.3 实验流程

#### 3.3.1 整体流程图

```
原始数据（6,662,647条）
    ↓
【数据抽样】分层抽样（12个月×2类别×5,000条）
    ↓
抽样数据（120,000条）
    ↓
【数据清洗】时间/空间/速度过滤
    ↓
清洗数据（58,405条）
    ↓
【特征工程】构造37个特征
    ↓
特征数据（58,405×37）
    ↓
【数据划分】80%训练集，20%测试集（分层划分）
    ↓
训练集（46,724条） | 测试集（11,681条）
    ↓                    ↓
【数据标准化】        【数据标准化】
StandardScaler         (使用训练集参数)
    ↓                    ↓
【模型训练】          【模型预测】
GridSearchCV           predict
3折交叉验证            ↓
    ↓               【性能评估】
【模型选择】        Accuracy, F1, 
最佳参数            Precision, Recall
    ↓
【最终模型】
    ↓
【结果分析】
特征重要性、混淆矩阵、因素影响
```
