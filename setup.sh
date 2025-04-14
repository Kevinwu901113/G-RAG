#!/bin/bash

echo "开始安装LightRAG环境..."

# 方法1：使用environment.yml文件创建环境
echo "方法1：使用environment.yml创建环境"
echo "运行: conda env create -f environment.yml"

# 方法2：手动创建环境
echo "方法2：手动创建环境"
echo "运行以下命令:"
echo "conda create -n LightRAG python=3.10.6 -y"
echo "conda activate LightRAG"
echo "pip install llama-index==0.12.9 openai==1.58.1 langchain==0.3.13 lightrag==0.1.0b6 lightrag-hku==1.0.1"
echo "pip install numpy pandas scipy scikit-learn matplotlib torch transformers"

# 方法3：使用requirements.txt创建环境（推荐）
echo "方法3：使用requirements.txt创建环境"
echo "运行以下命令:"
echo "conda create -n LightRAG python=3.10.6 -y"
echo "conda activate LightRAG"
echo "pip install -r requirements.txt"

# 创建环境的主要命令
create_env_yml() {
    conda env create -f environment.yml
}

create_env_req() {
    conda create -n LightRAG python=3.10.6 -y
    eval "$(conda shell.bash hook)"
    conda activate LightRAG
    pip install -r requirements.txt
}

# 检查环境是否已存在
if conda info --envs | grep -q "LightRAG"; then
    echo "LightRAG环境已存在。如果要重新创建，请先运行: conda remove -n LightRAG --all"
else
    echo "选择安装方法: "
    echo "1. 使用environment.yml (可能会有兼容性问题)"
    echo "2. 使用requirements.txt (推荐)"
    read method
    if [ "$method" == "1" ]; then
        echo "使用environment.yml创建环境..."
        create_env_yml
        echo "环境创建完成！使用 'conda activate LightRAG' 激活环境"
    elif [ "$method" == "2" ]; then
        echo "使用requirements.txt创建环境..."
        create_env_req
        echo "环境创建完成！使用 'conda activate LightRAG' 激活环境"
    else
        echo "已取消环境创建"
    fi
fi

echo "setup完成" 