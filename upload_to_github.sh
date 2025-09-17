#!/bin/bash
# GitHub上传脚本
# 使用方法: ./upload_to_github.sh <GitHub用户名> <仓库名>

if [ $# -ne 2 ]; then
    echo "使用方法: $0 <GitHub用户名> <仓库名>"
    echo "例如: $0 yourusername iTAC-AD"
    exit 1
fi

USERNAME=$1
REPO_NAME=$2
REPO_URL="https://github.com/${USERNAME}/${REPO_NAME}.git"

echo "准备上传到: ${REPO_URL}"

# 添加远程仓库
git remote add origin ${REPO_URL}

# 推送代码到GitHub
echo "正在推送代码到GitHub..."
git push -u origin main

echo "上传完成！"
echo "您可以在 https://github.com/${USERNAME}/${REPO_NAME} 查看您的仓库"
