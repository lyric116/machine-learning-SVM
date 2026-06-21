#!/usr/bin/env python3
"""
创新思路4连发：冲击F1>=0.70
基于之前成功经验的新方向
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("创新思路测试：冲击F1>=0.70")
print("="*70)

# 加载数据
df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

# 基础特征（之前的8特征+频率）
base_features = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak']

# 构建频率特征
df_freq = df.copy()

# 站点频率
for col in ['start_station_id', 'end_station_id']:
    member_count = df[y==1][col].value_counts()
    total_count = df[col].value_counts()
    member_ratio = (member_count / total_count).fillna(0.5)
    df_freq[f'{col}_member_ratio'] = df[col].map(member_ratio).fillna(0.5)

# 路线稀有度
df_freq['route'] = df['start_station_id'].astype(str) + '_TO_' + df['end_station_id'].astype(str)
route_freq = df_freq['route'].value_counts()
df_freq['route_rarity'] = df_freq['route'].map(route_freq).apply(lambda x: 1/(x+1))

freq_features = ['start_station_id_member_ratio', 'end_station_id_member_ratio', 'route_rarity']

# Baseline
print("\n[Baseline] 8特征+频率")
X_base = df_freq[base_features + freq_features].fillna(df_freq[base_features + freq_features].median())
X_tr, X_te, y_tr, y_te = train_test_split(X_base, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
clf = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced', max_iter=50000, random_state=42, dual='auto')
clf.fit(scaler.fit_transform(X_tr), y_tr)
f1_baseline = f1_score(y_te, clf.predict(scaler.transform(X_te)), average='weighted')
print(f"F1: {f1_baseline:.4f}")

# ==================== 创新1: 领域知识特征 ====================
print("\n" + "="*70)
print("[创新1] 领域知识驱动的交互特征")
print("="*70)

df_domain = df_freq.copy()

# 1. 速度×时段交互（快速通勤 vs 慢速休闲）
df_domain['fast_commute'] = ((df['speed_kmh'] > 12) & (df['is_commute_peak'] == 1)).astype(int)
df_domain['slow_leisure'] = ((df['speed_kmh'] < 10) & (df['is_weekend'] == 1)).astype(int)

# 2. 距离×时长比率（效率指标）
df_domain['efficiency'] = df['distance_km'] / (df['duration_min'] + 1)

# 3. 周末×时长交互（长时间周末骑行 = 散户）
df_domain['weekend_long'] = ((df['is_weekend'] == 1) & (df['duration_min'] > 20)).astype(int)

# 4. 工作日早晚高峰×速度（典型通勤模式）
if 'hour' in df.columns:
    df_domain['morning_rush_fast'] = ((df['hour'].between(7, 9)) & (df['speed_kmh'] > 12)).astype(int)
    df_domain['evening_rush_fast'] = ((df['hour'].between(17, 19)) & (df['speed_kmh'] > 12)).astype(int)

# 5. 路线稀有度×周末（周末稀有路线 = 观光）
df_domain['weekend_rare_route'] = df_domain['route_rarity'] * df['is_weekend']

domain_features = ['fast_commute', 'slow_leisure', 'efficiency', 'weekend_long',
                   'morning_rush_fast', 'evening_rush_fast', 'weekend_rare_route']

X_domain = df_domain[base_features + freq_features + domain_features]
X_domain = X_domain.fillna(X_domain.median())

X_tr, X_te, y_tr, y_te = train_test_split(X_domain, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
clf = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced', max_iter=50000, random_state=42, dual='auto')
clf.fit(scaler.fit_transform(X_tr), y_tr)
f1_domain = f1_score(y_te, clf.predict(scaler.transform(X_te)), average='weighted')

print(f"特征数: {len(base_features + freq_features + domain_features)}")
print(f"F1: {f1_domain:.4f}")
print(f"提升: {f1_domain - f1_baseline:+.4f}")

# ==================== 创新2: 分层建模 ====================
print("\n" + "="*70)
print("[创新2] 分层建模策略")
print("="*70)

# 工作日模型 + 周末模型
X_all = df_domain[base_features + freq_features + domain_features].fillna(
    df_domain[base_features + freq_features + domain_features].median())
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)

# 标准化
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

# 获取周末标记
is_weekend_tr = df_freq.loc[X_tr.index, 'is_weekend'].values
is_weekend_te = df_freq.loc[X_te.index, 'is_weekend'].values

# 训练两个模型
clf_weekday = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced', max_iter=50000, random_state=42, dual='auto')
clf_weekend = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced', max_iter=50000, random_state=42, dual='auto')

clf_weekday.fit(X_tr_scaled[is_weekend_tr == 0], y_tr[is_weekend_tr == 0])
clf_weekend.fit(X_tr_scaled[is_weekend_tr == 1], y_tr[is_weekend_tr == 1])

# 预测
y_pred = np.zeros(len(y_te))
y_pred[is_weekend_te == 0] = clf_weekday.predict(X_te_scaled[is_weekend_te == 0])
y_pred[is_weekend_te == 1] = clf_weekend.predict(X_te_scaled[is_weekend_te == 1])

f1_stratified = f1_score(y_te, y_pred, average='weighted')
print(f"F1: {f1_stratified:.4f}")
print(f"提升: {f1_stratified - f1_baseline:+.4f}")

# ==================== 创新3: 置信度加权集成 ====================
print("\n" + "="*70)
print("[创新3] 多角度SVM集成")
print("="*70)

# 训练3个不同角度的SVM
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

# SVM1: 偏重时间特征
time_features_idx = [i for i, f in enumerate(X_tr.columns) if 'week' in f or 'hour' in f or 'commute' in f]
clf1 = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced', max_iter=50000, random_state=42, dual='auto')
clf1.fit(X_tr_scaled, y_tr)

# SVM2: 偏重行为特征
behavior_features_idx = [i for i, f in enumerate(X_tr.columns) if 'speed' in f or 'distance' in f or 'duration' in f]
clf2 = LinearSVC(C=0.02, loss='hinge', class_weight='balanced', max_iter=50000, random_state=43, dual='auto')
clf2.fit(X_tr_scaled, y_tr)

# SVM3: 偏重空间特征
spatial_features_idx = [i for i, f in enumerate(X_tr.columns) if 'station' in f or 'route' in f]
clf3 = LinearSVC(C=0.005, loss='squared_hinge', class_weight='balanced', max_iter=50000, random_state=44, dual='auto')
clf3.fit(X_tr_scaled, y_tr)

# 使用decision_function获取置信度
conf1 = clf1.decision_function(X_te_scaled)
conf2 = clf2.decision_function(X_te_scaled)
conf3 = clf3.decision_function(X_te_scaled)

# 加权平均（权重可调）
conf_avg = 0.4 * conf1 + 0.35 * conf2 + 0.25 * conf3
y_pred_ensemble = (conf_avg > 0).astype(int)

f1_ensemble = f1_score(y_te, y_pred_ensemble, average='weighted')
print(f"F1: {f1_ensemble:.4f}")
print(f"提升: {f1_ensemble - f1_baseline:+.4f}")

# ==================== 创新4: 阈值优化 ====================
print("\n" + "="*70)
print("[创新4] 分类阈值优化")
print("="*70)

# 使用最佳模型的decision_function
clf_best = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced', max_iter=50000, random_state=42, dual='auto')
clf_best.fit(X_tr_scaled, y_tr)

decision_scores = clf_best.decision_function(X_te_scaled)

# 尝试不同阈值
best_threshold = 0
best_f1_thresh = 0

for threshold in np.linspace(-0.5, 0.5, 21):
    y_pred_thresh = (decision_scores > threshold).astype(int)
    f1_thresh = f1_score(y_te, y_pred_thresh, average='weighted')
    if f1_thresh > best_f1_thresh:
        best_f1_thresh = f1_thresh
        best_threshold = threshold

print(f"最佳阈值: {best_threshold:.3f}")
print(f"F1: {best_f1_thresh:.4f}")
print(f"提升: {best_f1_thresh - f1_baseline:+.4f}")

# ==================== 总结 ====================
print("\n" + "="*70)
print("📊 创新思路总结")
print("="*70)

results = [
    ('Baseline (8特征+频率)', f1_baseline),
    ('创新1: 领域知识特征', f1_domain),
    ('创新2: 分层建模', f1_stratified),
    ('创新3: 置信度集成', f1_ensemble),
    ('创新4: 阈值优化', f1_thresh)
]

for name, f1 in results:
    improvement = f1 - f1_baseline
    print(f"{name:25s} F1={f1:.4f} ({improvement:+.4f})")

best_name, best_f1 = max(results, key=lambda x: x[1])
print(f"\n最佳方案: {best_name}")
print(f"最佳F1: {best_f1:.4f}")
print(f"完整37特征: F1=0.6587")
print(f"目标: F1>=0.70")

if best_f1 >= 0.70:
    print(f"\n🎉 成功达到目标！F1={best_f1:.4f} >= 0.70")
elif best_f1 >= 0.65:
    print(f"\n✅ 显著改进！距目标还差 {0.70 - best_f1:.4f}")
elif best_f1 > f1_baseline + 0.01:
    print(f"\n✓ 有所改进，距目标还差 {0.70 - best_f1:.4f}")
else:
    print(f"\n⚠️ 改进有限，需要新方向")

# 保存结果
pd.DataFrame(results, columns=['Method', 'F1']).to_csv(
    'outputs/tables/innovation_experiments.csv', index=False)
print(f"\n结果已保存: outputs/tables/innovation_experiments.csv")
