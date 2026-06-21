#!/usr/bin/env python3
"""
汇总所有实验结果
"""

import pandas as pd
import glob

print("="*70)
print("所有实验结果汇总")
print("="*70)

# 收集所有结果文件
result_files = glob.glob('outputs/tables/*results.csv')

all_results = []

for file in result_files:
    try:
        df = pd.read_csv(file)
        if 'f1' in df.columns or 'test_f1' in df.columns:
            all_results.append(df)
    except:
        pass

if all_results:
    results_df = pd.concat(all_results, ignore_index=True)

    # 提取F1列
    if 'test_f1' in results_df.columns:
        results_df['f1_score'] = results_df['test_f1']
    elif 'f1' in results_df.columns:
        results_df['f1_score'] = results_df['f1']

    # 按F1排序
    results_df = results_df.sort_values('f1_score', ascending=False)

    print("\n所有实验结果（按F1降序）：")
    print("="*70)

    for idx, row in results_df.iterrows():
        method = row.get('method', 'Unknown')
        f1 = row.get('f1_score', 0)
        print(f"{method:40s} F1 = {f1:.4f}")

    print("\n" + "="*70)

    best = results_df.iloc[0]
    print(f"最佳方案: {best.get('method', 'Unknown')}")
    print(f"最佳F1:   {best.get('f1_score', 0):.4f}")
    print(f"目标F1:   0.7000")

    best_f1 = best.get('f1_score', 0)

    if best_f1 >= 0.70:
        print(f"\n🎉🎉🎉 达到目标！F1 = {best_f1:.4f} >= 0.70")
    else:
        print(f"\n差距: {0.70 - best_f1:.4f} ({(0.70 - best_f1)/0.70*100:.1f}%)")

    # 保存汇总
    results_df.to_csv('outputs/tables/all_experiments_summary.csv', index=False)
    print(f"\n汇总已保存: outputs/tables/all_experiments_summary.csv")
else:
    print("未找到结果文件")
