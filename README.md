# LightRAG 项目环境安装指南

本文档提供了复制LightRAG项目环境的详细步骤。

## 环境要求

- Anaconda 或 Miniconda
- Python 3.10.6

## 安装方法

### 方法1：使用环境配置文件（推荐）

这是最简单的方法，可以完全复制原始环境:

```bash
# 克隆仓库
git clone [项目URL]
cd [项目目录]

# 创建并激活环境
conda env create -f environment.yml
conda activate LightRAG
```

### 方法2：使用脚本安装

提供了一个自动化脚本帮助安装:

```bash
# 添加执行权限
chmod +x setup.sh

# 运行安装脚本
./setup.sh
```

### 方法3：手动安装主要依赖

如果您不需要完全相同的环境，可以只安装核心依赖:

```bash
# 创建Python 3.10.6的conda环境
conda create -n LightRAG python=3.10.6 -y
conda activate LightRAG

# 安装核心依赖
pip install llama-index==0.12.9 
pip install openai==1.58.1 
pip install langchain==0.3.13 
pip install lightrag==0.1.0b6 
pip install lightrag-hku==1.0.1

# 安装其他常用依赖
pip install numpy pandas scikit-learn matplotlib torch transformers
```

## 验证安装

安装完成后，可以运行以下命令验证环境:

```bash
# 激活环境
conda activate LightRAG

# 验证Python版本
python --version  # 应显示 Python 3.10.6

# 验证关键包
python -c "import lightrag; print(f'lightrag版本: {lightrag.__version__}')"
python -c "import llama_index; print(f'llama_index版本: {llama_index.__version__}')"
```

## 注意事项

- 环境配置文件中使用了中国镜像源，国际用户可能需要修改
- 完整环境较大，安装可能需要一些时间
- 如果遇到包冲突，建议使用方法1或方法2进行安装

## 故障排除

如果安装过程中遇到问题:

1. 确保已正确安装Anaconda或Miniconda
2. 检查是否有足够的磁盘空间
3. 如果某些包安装失败，可尝试单独安装它们
4. 对于网络问题，可以尝试使用国内镜像源 