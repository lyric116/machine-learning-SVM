#!/usr/bin/env python3
"""
单特征SVM分类实验
测试每个特征单独的分类能力
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

print("单特征SVM分类实验")
print("="*60)

# 加载已清洗的数据
csv_path = Path('outputs/processed/cleaned_stratified_sample.csv')
if not csv_path.exists():
    print("未找到清洗后数据，请先运行主程序")
    exit(1)

df = pd.read_csv(csv_path)
print(f"样本数: {len(df)}")

# 准备目标变量
y = (df['member_casual'] == 'member').astype(int)

# 获取所有特征（排除ID和目标）
exclude_cols = ['ride_id', 'member_casual', 'started_at', 'ended_at']
all_features = [c for c in df.columns if c not in exclude_cols]

print(f"可用特征: {len(all_features)}")
print("\n开始测试每个特征...")

results = []

for feat in all_features:
    try:
        # 准备单特征数据
        X = df[[feat]].copy()

        # 处理类别特征
        if X[feat].dtype == 'object':
            le = LabelEncoder()
            X[feat] = le.fit_transform(X[feat].fillna('missing'))
        else:
            X = X.fillna(X.median())

        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练SVM
        clf = LinearSVC(C=0.02, class_weight='balanced',
                       max_iter=10000, random_state=42, dual='auto')
        clf.fit(X_train_scaled, y_train)

        # 评估
        y_pred = clf.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)

        results.append({
            'feature': feat,
            'f1': f1,
            'accuracy': acc,
            'type': 'categorical' if df[feat].dtype == 'object' else 'numeric'
        })

        print(f"  {feat:30s} F1={f1:.4f} Acc={acc:.4f}")

    except Exception as e:
        print(f"  {feat:30s} 跳过 ({str(e)[:30]})")

# 排序结果
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1', ascending=False)

print("\n" + "="*60)
print("单特征F1排名 (Top 15):")
print("="*60)

for i, row in results_df.head(15).iterrows():
    print(f"{row['feature']:30s} F1={row['f1']:.4f} ({row['type']})")

# 保存结果
output_path = Path('outputs/tables/single_feature_results.csv')
results_df.to_csv(output_path, index=False)
print(f"\n完整结果已保存: {output_path}")

# 关键发现
best = results_df.iloc[0]
baseline_f1 = 0.6587

print("\n" + "="*60)
print("关键发现:")
print("="*60)
print(f"最佳单特征: {best['feature']}")
print(f"单特征F1:   {best['f1']:.4f}")
print(f"完整模型F1: {baseline_f1:.4f}")
print(f"差距:       {baseline_f1 - best['f1']:.4f} ({(baseline_f1-best['f1'])/baseline_f1*100:.1f}%)")
print(f"\n结论: {'单特征已接近完整模型' if best['f1'] > 0.60 else '需要多特征协同'}")
