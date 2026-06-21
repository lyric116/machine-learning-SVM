#!/usr/bin/env python3
"""
最小特征集优化实验
寻找最少特征达到最高F1的组合
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from itertools import combinations

print("最小特征集优化实验")
print("="*70)

# 加载数据
df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

# 候选特征池（基于单特征测试+领域知识）
candidate_features = {
    # 核心数值特征
    'speed_kmh': 'numeric',
    'duration_min': 'numeric',
    'distance_km': 'numeric',
    'weekday': 'numeric',
    'hour': 'numeric',
    'is_weekend': 'numeric',
    'is_commute_peak': 'numeric',
    'start_lat': 'numeric',
    'start_lng': 'numeric',
    'end_lat': 'numeric',
    'end_lng': 'numeric',
    # 类别特征
    'rideable_type': 'categorical',
    'start_station_id': 'categorical',
    'end_station_id': 'categorical',
}

print(f"候选特征池: {len(candidate_features)}个")
print()

def evaluate_feature_set(features, df, y):
    """评估特征组合"""
    X = df[features].copy()

    # 处理类别特征
    for col in features:
        if candidate_features[col] == 'categorical':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('missing'))
        else:
            X[col] = X[col].fillna(X[col].median())

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 使用最佳配置训练
    clf = LinearSVC(C=0.003, loss='hinge', class_weight='balanced',
                   max_iter=50000, random_state=42, dual='auto')
    clf.fit(X_train_scaled, y_train)

    # 评估
    y_pred = clf.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)

    return f1, acc

# ==================== 策略1: 贪心搜索 ====================
print("策略1: 贪心特征选择")
print("-"*70)

selected = []
available = list(candidate_features.keys())
best_f1 = 0

for step in range(1, 11):  # 最多10个特征
    best_feature = None
    best_step_f1 = 0

    for feat in available:
        test_features = selected + [feat]
        try:
            f1, acc = evaluate_feature_set(test_features, df, y)
            if f1 > best_step_f1:
                best_step_f1 = f1
                best_feature = feat
        except:
            continue

    if best_feature:
        selected.append(best_feature)
        available.remove(best_feature)
        print(f"步骤{step}: +{best_feature:20s} F1={best_step_f1:.4f}")

        if best_step_f1 > best_f1:
            best_f1 = best_step_f1
    else:
        break

print()
print(f"贪心搜索最佳: {len(selected)}个特征, F1={best_f1:.4f}")
print(f"特征: {', '.join(selected)}")

# ==================== 策略2: 手工精选组合 ====================
print()
print("策略2: 专家精选组合")
print("-"*70)

expert_combinations = {
    '核心3特征': ['speed_kmh', 'weekday', 'duration_min'],
    '时空5特征': ['speed_kmh', 'duration_min', 'weekday', 'start_lat', 'end_lat'],
    '增强7特征': ['speed_kmh', 'duration_min', 'weekday', 'is_weekend',
                 'is_commute_peak', 'start_lat', 'end_lat'],
    '站点8特征': ['speed_kmh', 'duration_min', 'weekday', 'is_commute_peak',
                 'start_station_id', 'end_station_id', 'start_lat', 'end_lat'],
}

expert_results = []

for name, features in expert_combinations.items():
    try:
        f1, acc = evaluate_feature_set(features, df, y)
        expert_results.append({
            'name': name,
            'n_features': len(features),
            'features': features,
            'f1': f1,
            'accuracy': acc
        })
        print(f"{name:15s} ({len(features)}特征): F1={f1:.4f}")
    except Exception as e:
        print(f"{name:15s} 失败: {str(e)[:40]}")

# ==================== 策略3: 网格搜索最佳C值 ====================
print()
print("策略3: 精细调参（基于最佳特征集）")
print("-"*70)

# 使用贪心找到的最佳特征集
best_features = selected[:7]  # 取前7个
X = df[best_features].copy()

for col in best_features:
    if candidate_features[col] == 'categorical':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna('missing'))
    else:
        X[col] = X[col].fillna(X[col].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 精细网格搜索
param_grid = {
    'C': [0.001, 0.002, 0.003, 0.005, 0.007, 0.01],
    'loss': ['hinge', 'squared_hinge'],
    'class_weight': ['balanced']
}

grid = GridSearchCV(
    LinearSVC(max_iter=50000, random_state=42, dual='auto'),
    param_grid,
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1
)

print(f"特征: {', '.join(best_features)}")
print(f"网格搜索中...")

grid.fit(X_train_scaled, y_train)
y_pred = grid.predict(X_test_scaled)
final_f1 = f1_score(y_test, y_pred, average='weighted')
final_acc = accuracy_score(y_test, y_pred)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳CV F1: {grid.best_score_:.4f}")
print(f"测试集F1: {final_f1:.4f}")
print(f"测试集Acc: {final_acc:.4f}")

# ==================== 汇总 ====================
print()
print("="*70)
print("最终汇总")
print("="*70)

results_summary = {
    '单特征(speed)': {'n': 1, 'f1': 0.5672},
    'Top-3特征': {'n': 3, 'f1': 0.5833},
    f'贪心{len(selected)}特征': {'n': len(selected), 'f1': best_f1},
    f'精调{len(best_features)}特征': {'n': len(best_features), 'f1': final_f1},
    '完整37特征': {'n': 37, 'f1': 0.6587},
}

for name, result in results_summary.items():
    pct = result['f1'] / 0.6587 * 100
    print(f"{name:20s} F1={result['f1']:.4f} ({pct:.1f}% of best)")

print()
print("💡 关键发现:")

if final_f1 >= 0.63:
    print(f"✨ 仅用{len(best_features)}个特征就达到了F1={final_f1:.4f}！")
    print(f"   相比完整模型(0.6587)仅差 {0.6587-final_f1:.4f}")
elif final_f1 >= 0.60:
    print(f"✓ {len(best_features)}个特征可达F1={final_f1:.4f}")
    print(f"  还差 {0.6587-final_f1:.4f} 需要高级特征工程")
else:
    print(f"  {len(best_features)}个特征F1={final_f1:.4f}")
    print(f"  完整37特征的优势明显 (+{0.6587-final_f1:.4f})")

# 保存结果
results_df = pd.DataFrame([
    {'strategy': 'Greedy', 'n_features': len(selected), 'features': ','.join(selected), 'f1': best_f1},
    {'strategy': 'Tuned', 'n_features': len(best_features), 'features': ','.join(best_features),
     'f1': final_f1, 'best_params': str(grid.best_params_)},
])

results_df.to_csv('outputs/tables/minimal_features_optimization.csv', index=False)
print()
print("结果已保存: outputs/tables/minimal_features_optimization.csv")
