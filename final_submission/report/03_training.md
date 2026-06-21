#### 3.3.2 模型训练与验证

**训练集与测试集划分**：
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80%训练，20%测试
    random_state=42,     # 固定随机种子，保证可复现
    stratify=y          # 分层划分，保持类别比例
)
```

**超参数搜索**：
使用GridSearchCV进行参数优化
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'C': [0.02, 0.03, 0.05, 0.1, 0.25, 0.5],
    'loss': ['hinge', 'squared_hinge'],
    'class_weight': [None, 'balanced']
}

clf = LinearSVC(max_iter=50000, random_state=42, dual='auto')

grid = GridSearchCV(
    clf,
    param_grid,
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)
```

**交叉验证策略**：
- 使用3折分层交叉验证（StratifiedKFold）
- 保证每折中正负样本比例一致
- 评估指标：加权F1分数（f1_weighted）

**评估指标**：

1. **准确率（Accuracy）**：
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **精确率（Precision）**：
$$Precision = \frac{TP}{TP + FP}$$

3. **召回率（Recall）**：
$$Recall = \frac{TP}{TP + FN}$$

4. **F1分数（F1-Score）**：
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

其中TP、TN、FP、FN分别表示真正例、真负例、假正例、假负例。

**加权F1**：考虑类别不平衡，使用各类别F1的加权平均
$$F1_{weighted} = \sum_{i} w_i \times F1_i$$
其中$w_i$是类别$i$的样本占比。
