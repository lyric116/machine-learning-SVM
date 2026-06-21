#!/bin/bash
# 解决GitHub大文件问题的脚本

echo "=========================================="
echo "解决GitHub大文件推送失败"
echo "=========================================="
echo ""

echo "问题的大文件："
echo "  1. outputs/processed/enhanced_sample.csv (133.37 MB)"
echo "  2. f1_optimization_clean/outputs/cache/cleaned_sample_20000.csv (542.02 MB)"
echo "  3. f1_optimization_clean/outputs/models/best_member_casual_model.joblib (742.47 MB)"
echo "  4. f1_target_optimization/outputs_natural/natural_cleaned_20000.csv (121.99 MB)"
echo ""

echo "这些文件应该被.gitignore排除，但在历史commit中已经存在"
echo ""

# 方案1：从Git历史中完全删除这些大文件（推荐）
echo "【方案1：从Git历史中删除大文件】（推荐）"
echo ""
echo "执行以下命令："
echo ""
cat << 'SOLUTION1'
# 1. 从Git历史中删除大文件
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch \
    outputs/processed/enhanced_sample.csv \
    f1_optimization_clean/outputs/cache/cleaned_sample_20000.csv \
    f1_optimization_clean/outputs/models/best_member_casual_model.joblib \
    f1_target_optimization/outputs_natural/natural_cleaned_20000.csv' \
  --prune-empty --tag-name-filter cat -- --all

# 2. 清理和优化
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 3. 强制推送（因为改写了历史）
git push origin main --force

SOLUTION1

echo ""
echo "=========================================="
echo ""

# 方案2：使用BFG Repo-Cleaner（更快，需要安装）
echo "【方案2：使用BFG Repo-Cleaner】（更快，但需要安装工具）"
echo ""
cat << 'SOLUTION2'
# 1. 安装BFG（如果没有）
# Ubuntu/Debian: sudo apt install bfg
# 或下载: wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# 2. 删除大于100MB的文件
java -jar bfg.jar --strip-blobs-bigger-than 100M .

# 3. 清理
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 4. 推送
git push origin main --force

SOLUTION2

echo ""
echo "=========================================="
echo ""

# 方案3：简单方案 - 重置到删除大文件前的状态
echo "【方案3：最简单的方案】（如果前面的commit不重要）"
echo ""
cat << 'SOLUTION3'
# 1. 查看commit历史，找到添加大文件之前的commit
git log --oneline

# 2. 硬重置到那个commit（假设是91f2153）
git reset --hard 91f2153

# 3. 确保.gitignore正确（已经有了新的.gitignore）
cat .gitignore

# 4. 重新添加文件
git add .

# 5. 提交
git commit -m "Clean repository structure with proper .gitignore

- Remove large data files and model files
- Keep only essential submission files
- F1 score: 64.11%, complete project

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"

# 6. 强制推送
git push origin main --force

SOLUTION3

echo ""
echo "=========================================="
echo "推荐：使用方案3（最简单）"
echo "=========================================="
