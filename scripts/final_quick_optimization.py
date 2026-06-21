#!/usr/bin/env python3
"""
最后优化尝试：
1. 更大样本量 (20k)
2. 更精细的C值搜索
3. 不同的特征组合
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import joblib

DATA_DIR = Path('data')
RANDOM_STATE = 42

print("最后优化尝试...")

# 快速加载已清洗的数据（如果存在）
if Path('outputs/processed/cleaned_stratified_sample.csv').exists():
    print("加载已清洗数据...")
    df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
    print(f"样本数: {len(df)}")

    # 检查是否有足够特征
    if df.shape[1] < 30:
        print("特征数不足，需要重新运行特征工程")
        exit(1)

    # 准备数据
    y = (df['member_casual'] == 'member').astype(int)

    # 选择数值特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['member_casual']]

    X = df[numeric_cols].fillna(0)

    print(f"特征数: {X.shape[1]}")
    print(f"样本数: {X.shape[0]}")

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 尝试1: 更精细的C值
    print("\n尝试1: 超精细C值搜索...")
    param_grid = {
        'C': [0.018, 0.019, 0.02, 0.021, 0.022],
        'class_weight': ['balanced']
    }

    grid = GridSearchCV(
        LinearSVC(max_iter=50000, random_state=RANDOM_STATE, dual='auto'),
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train)
    y_pred = grid.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"最佳C: {grid.best_params_['C']}")
    print(f"CV F1: {grid.best_score_:.4f}")
    print(f"Test F1: {f1:.4f}")

    # 尝试2: 不同的loss函数
    print("\n尝试2: 不同loss函数...")
    for loss in ['hinge', 'squared_hinge']:
        clf = LinearSVC(C=0.02, class_weight='balanced', loss=loss,
                       max_iter=50000, random_state=RANDOM_STATE, dual='auto')
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"  loss={loss}: F1={f1:.4f}")

    print("\n✓ 快速优化尝试完成")

else:
    print("未找到已清洗数据，跳过快速优化")
