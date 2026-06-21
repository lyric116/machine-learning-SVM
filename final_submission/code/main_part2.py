# ============================================================================
# 6. 模型评估
# ============================================================================
print("\n[6/6] 模型评估...")

# 在测试集上预测
y_pred = best_model.predict(X_test_scaled)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*70)
print("最终结果")
print("="*70)

print(f"\n准确率 (Accuracy): {accuracy:.4f}")
print(f"F1分数 (F1-Score): {f1:.4f}")

print("\n分类报告:")
print(classification_report(
    y_test, y_pred,
    target_names=['Casual', 'Member'],
    digits=4
))

print("混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(f"{'':12s} 预测Casual  预测Member")
print(f"实际Casual  {cm[0,0]:10d}  {cm[0,1]:10d}")
print(f"实际Member  {cm[1,0]:10d}  {cm[1,1]:10d}")

# ============================================================================
# 7. 保存模型
# ============================================================================
print("\n" + "="*70)
print("保存模型...")
print("="*70)

# 保存模型和scaler
joblib.dump(best_model, 'final_submission/models/svm_model.joblib')
joblib.dump(scaler, 'final_submission/models/scaler.joblib')

print("  模型已保存: final_submission/models/svm_model.joblib")
print("  标准化器已保存: final_submission/models/scaler.joblib")

# 保存特征名
feature_names = pd.DataFrame({'feature': feature_cols})
feature_names.to_csv('final_submission/models/feature_names.csv', index=False)
print("  特征名已保存: final_submission/models/feature_names.csv")

# 保存结果
results = {
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'best_params': grid_search.best_params_,
    'cv_f1': float(grid_search.best_score_),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'n_features': len(feature_cols)
}

import json
with open('final_submission/models/results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("  结果已保存: final_submission/models/results.json")

print("\n" + "="*70)
print("实验完成！")
print("="*70)
print(f"\n核心成果:")
print(f"  - 测试集准确率: {accuracy:.2%}")
print(f"  - 测试集F1分数: {f1:.2%}")
print(f"  - 训练样本数: {len(X_train):,}")
print(f"  - 测试样本数: {len(X_test):,}")
print(f"  - 使用特征数: {len(feature_cols)}")
print(f"  - 最佳参数: C={grid_search.best_params_['C']}, "
      f"loss={grid_search.best_params_['loss']}, "
      f"class_weight={grid_search.best_params_['class_weight']}")
