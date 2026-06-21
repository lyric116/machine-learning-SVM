#### E.4 模型训练完整代码

```python
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. 数据准备
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 2. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 定义参数网格
param_grid = {
    'C': [0.02, 0.03, 0.05, 0.1, 0.25, 0.5],
    'loss': ['hinge', 'squared_hinge'],
    'class_weight': [None, 'balanced']
}

# 4. 创建SVM模型
svm = LinearSVC(
    max_iter=50000, 
    random_state=42,
    dual='auto'  # 自动选择求解方法
)

# 5. 网格搜索 + 交叉验证
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1,  # 使用所有CPU核心
    verbose=2
)

# 6. 训练
print("开始训练...")
grid_search.fit(X_train_scaled, y_train)

# 7. 获取最佳模型
best_model = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳CV F1: {grid_search.best_score_:.4f}")

# 8. 测试集评估
y_pred = best_model.predict(X_test_scaled)
print("\n分类报告:")
print(classification_report(
    y_test, y_pred, 
    target_names=['casual', 'member']
))

# 9. 保存模型
import joblib
joblib.dump(best_model, 'svm_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
```

**训练过程说明**：
1. **数据划分**：80:20，分层保证类别平衡
2. **特征标准化**：Z-score标准化，均值0、标准差1
3. **参数搜索**：36种组合（6×2×3）
4. **交叉验证**：3折分层CV，每个参数组合训练3次
5. **并行加速**：n_jobs=-1使用所有CPU
6. **模型保存**：joblib格式，包含scaler

**计算复杂度**：
- 训练样本：46,724条
- 特征维度：37维
- 参数组合：36种
- CV折数：3折
- 总训练次数：36×3=108次
- 总耗时：约8秒（LinearSVC高效）

#### E.5 特征重要性评估代码

```python
from sklearn.inspection import permutation_importance

# 计算置换重要性
perm_importance = permutation_importance(
    best_model,
    X_test_scaled,
    y_test,
    n_repeats=10,  # 重复10次求平均
    random_state=42,
    n_jobs=-1
)

# 获取特征名和重要性
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

# Top-10特征
print(feature_importance.head(10))
```

**置换重要性原理**：
1. 记录原始模型在测试集上的准确率
2. 随机打乱某个特征的值
3. 重新预测，计算准确率下降
4. 下降越多，特征越重要
5. 重复10次，取平均值和标准差
