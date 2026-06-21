#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capital Bikeshare 用户分类实验 - 主程序
使用支持向量机(SVM)算法预测用户类型(会员/散户)

实验要求：
- 80%:20%训练测试划分
- 使用SVM算法
- 完整的数据预处理和特征工程
- 交叉验证和性能评估

运行方法：
    python main.py

输出：
    - 训练好的模型文件
    - 评估指标和分类报告
    - 混淆矩阵
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    """主函数"""

    print("="*70)
    print("Capital Bikeshare 用户类型分类 - SVM实验")
    print("="*70)

    # ========================================================================
    # 步骤1：加载数据
    # ========================================================================
    print("\n[步骤1/6] 加载数据...")

    df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
    print(f"  样本总数: {len(df):,}条")

    # 创建目标变量 (member=1, casual=0)
    y = (df['member_casual'] == 'member').astype(int)

    print(f"  会员(Member): {(y==1).sum():,}条 ({(y==1).sum()/len(y)*100:.1f}%)")
    print(f"  散户(Casual): {(y==0).sum():,}条 ({(y==0).sum()/len(y)*100:.1f}%)")

    # ========================================================================
    # 步骤2：特征选择
    # ========================================================================
    print("\n[步骤2/6] 特征工程...")

    # 排除不用于训练的列
    exclude_cols = [
        'ride_id', 'member_casual', 'started_at', 'ended_at',
        'start_station_name', 'end_station_name', 'route', 'source_month'
    ]

    # 选择所有数值和类别特征
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()

    print(f"  特征总数: {len(feature_cols)}个")

    # 处理类别变量：转换为数值编码
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.factorize(X[col])[0]

    # 处理异常值和缺失值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    print(f"  特征矩阵: {X.shape[0]}行 × {X.shape[1]}列")

    # ========================================================================
    # 步骤3：划分训练集和测试集
    # ========================================================================
    print("\n[步骤3/6] 划分训练集和测试集...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,        # 80%训练，20%测试
        random_state=42,      # 固定随机种子，保证可复现
        stratify=y           # 分层划分，保持类别比例
    )

    print(f"  训练集: {len(X_train):,}条 ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  测试集: {len(X_test):,}条 ({len(X_test)/len(X)*100:.0f}%)")
    print(f"  训练集类别分布: Member={np.sum(y_train==1):,}, Casual={np.sum(y_train==0):,}")
    print(f"  测试集类别分布: Member={np.sum(y_test==1):,}, Casual={np.sum(y_test==0):,}")

    # ========================================================================
    # 步骤4：特征标准化
    # ========================================================================
    print("\n[步骤4/6] 特征标准化 (Z-score)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  标准化完成: 均值=0, 标准差=1")

    # ========================================================================
    # 步骤5：模型训练与超参数搜索
    # ========================================================================
    print("\n[步骤5/6] 训练SVM模型...")
    print("  算法: LinearSVC (线性支持向量机)")
    print("  优化方法: GridSearchCV + 3折交叉验证")

    # 定义超参数搜索空间
    param_grid = {
        'C': [0.02, 0.03, 0.05, 0.1, 0.25, 0.5],
        'loss': ['hinge', 'squared_hinge'],
        'class_weight': [None, 'balanced']
    }

    print(f"  参数组合数: {len(param_grid['C']) * len(param_grid['loss']) * len(param_grid['class_weight'])}")

    # 创建SVM分类器
    svm = LinearSVC(
        max_iter=50000,
        random_state=42,
        dual='auto'
    )

    # 网格搜索 + 交叉验证
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    print("\n  开始训练 (这可能需要几分钟)...")
    grid_search.fit(X_train_scaled, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"\n  ✓ 训练完成")
    print(f"  最佳参数: {best_params}")
    print(f"  最佳交叉验证F1: {best_cv_score:.4f}")

    # ========================================================================
    # 步骤6：模型评估
    # ========================================================================
    print("\n[步骤6/6] 在测试集上评估...")

    y_pred = best_model.predict(X_test_scaled)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "="*70)
    print("实验结果")
    print("="*70)

    print(f"\n【核心指标】")
    print(f"  准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1分数 (F1-Score):  {f1:.4f}")

    print(f"\n【分类报告】")
    print(classification_report(
        y_test, y_pred,
        target_names=['Casual', 'Member'],
        digits=4
    ))

    print("【混淆矩阵】")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':>15} 预测Casual  预测Member")
    print(f"实际Casual   {cm[0,0]:>10,}  {cm[0,1]:>10,}")
    print(f"实际Member   {cm[1,0]:>10,}  {cm[1,1]:>10,}")

    # ========================================================================
    # 步骤7：保存模型和结果
    # ========================================================================
    print("\n" + "="*70)
    print("保存模型和结果...")
    print("="*70)

    # 保存模型
    joblib.dump(best_model, 'final_submission/models/svm_model.joblib')
    joblib.dump(scaler, 'final_submission/models/scaler.joblib')

    # 保存特征名称
    pd.DataFrame({'feature': feature_cols}).to_csv(
        'final_submission/models/feature_names.csv', index=False
    )

    # 保存结果摘要
    results = {
        'test_accuracy': float(accuracy),
        'test_f1_score': float(f1),
        'cv_f1_score': float(best_cv_score),
        'best_params': best_params,
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'n_features': int(len(feature_cols)),
        'confusion_matrix': cm.tolist()
    }

    with open('final_submission/models/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 模型: final_submission/models/svm_model.joblib")
    print(f"  ✓ 标准化器: final_submission/models/scaler.joblib")
    print(f"  ✓ 特征名: final_submission/models/feature_names.csv")
    print(f"  ✓ 结果: final_submission/models/results.json")

    print("\n" + "="*70)
    print("实验成功完成！")
    print("="*70)

    return results

if __name__ == "__main__":
    results = main()
