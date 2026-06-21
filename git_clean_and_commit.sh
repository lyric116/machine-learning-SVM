#!/bin/bash
# Git仓库清理和重新提交脚本

echo "=========================================="
echo "Git 仓库清理和重新提交"
echo "=========================================="
echo ""

# 第一步：从Git索引中移除所有文件（不删除本地文件）
echo "[1/6] 移除所有已追踪的文件..."
git rm -r --cached .
echo "✓ 已从Git索引移除所有文件"
echo ""

# 第二步：添加更新后的.gitignore
echo "[2/6] 添加新的.gitignore..."
git add .gitignore
echo "✓ .gitignore已添加"
echo ""

# 第三步：根据新的.gitignore重新添加文件
echo "[3/6] 根据新的.gitignore重新添加文件..."
git add .
echo "✓ 文件已重新添加"
echo ""

# 第四步：查看将要提交的文件
echo "[4/6] 查看将要提交的文件..."
echo ""
git status
echo ""

# 第五步：确认
echo "=========================================="
echo "准备提交的文件已列出"
echo "=========================================="
echo ""
echo "是否继续提交？(y/n)"
read -p "> " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo ""
    echo "[5/6] 提交更改..."
    git commit -m "Clean up repository and reorganize project structure

- Remove unnecessary files (old markdown docs, logs, experimental archives)
- Add comprehensive .gitignore
- Keep only essential submission files
- Organize into FINAL_SUBMISSION_最终提交/ directory

Project summary:
- F1 score: 64.11%
- 3 major findings: frequency features, minimal feature set, linear kernel optimal
- Complete report (~4000 words)
- 60 files in clean submission structure

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"

    echo ""
    echo "[6/6] 查看提交结果..."
    git log --oneline -3
    echo ""
    echo "=========================================="
    echo "✅ 完成！"
    echo "=========================================="
    echo ""
    echo "下一步："
    echo "  git push origin main"
else
    echo ""
    echo "❌ 已取消提交"
    echo "你可以使用 'git status' 查看状态"
fi
