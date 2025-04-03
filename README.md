# Graph-Enhanced LightRAG

## 项目概述

Graph-Enhanced LightRAG是一个融合图结构嵌入的结构化检索生成框架，它在传统RAG（检索增强生成）的基础上引入了图神经网络（GNN）进行结构化嵌入，使得每个节点的表示向量能融合上下文邻居信息，从而构建结构感知语义空间，提升检索质量和生成能力。

### 核心创新点

- **图结构感知嵌入**：使用图神经网络（如GAT、GraphSAGE等）对文档图中的节点进行结构感知的嵌入
- **混合检索策略**：基于图嵌入向量进行enhanced retrieval，结合原始embedding检索
- **孤立节点连接**：通过相似度和主题关联自动连接孤立节点，提高图的连通性

## 系统架构

系统由以下几个核心模块组成：

1. **文档处理模块**：负责将原始文档切分成适当大小的文本块
2. **图构建模块**：基于文本相似度、实体共现等策略构建文档图
3. **图嵌入模块**：使用GNN生成结构感知的节点表示
4. **检索模块**：支持原始嵌入和图增强嵌入的混合检索
5. **生成模块**：使用检索到的文档生成最终答案

## 项目结构

```
GraphLightRAG/
├── data/
│   └── raw_docs/                 # 原始文档
│   └── chunks.json               # 文档切分结果
├── graph/
│   └── build_graph.py            # 图构建脚本
│   └── gnn_embed.py              # 图嵌入模型定义
├── retrieval/
│   └── faiss_index.py            # FAISS 索引构建与检索
├── generator/
│   └── generate_answer.py        # 使用 FiD/T5 生成答案
├── utils/
│   └── text_splitter.py          # 文本切分
│   └── embedding.py              # 使用 huggingface 获取文本嵌入
├── run_pipeline.py               # 主流程脚本（读取问题 -> 检索 -> 生成）
├── config.yaml                   # 配置文件
└── requirements.txt              # 依赖包列表
```

## 安装与配置

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA（可选，用于GPU加速）

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/yourusername/Graph-Enhanced-LightRAG.git
cd Graph-Enhanced-LightRAG
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 下载必要的模型（如果需要）

```bash
python -m spacy download en_core_web_sm
```

## 使用方法

### 1. 准备数据

将您的文档放入 `data/raw_docs/` 目录中。支持的格式包括：txt, md, html, htm, xml, json, csv, docx。

### 2. 配置系统

编辑 `config.yaml` 文件，根据需要调整参数：

- 文档处理参数（块大小、重叠大小等）
- 图构建参数（相似度阈值、实体重叠阈值等）
- 嵌入和GNN参数（模型类型、维度等）
- 检索和生成参数（top-k、模型名称等）

### 3. 运行系统

#### 处理文档并构建索引

```bash
python run_pipeline.py --reprocess
```

#### 交互式问答

```bash
python run_pipeline.py --interactive
```

#### 单次查询

```bash
python run_pipeline.py --query "你的问题"
```

#### 评估模式

```bash
python run_pipeline.py --evaluate --test_data path/to/test_data.json
```

## 实验与评估

系统支持以下评估指标：

- **检索质量**：Recall@k、Precision@k
- **生成质量**：ROUGE、BERTScore
- **结构利用**：孤立节点参与率、平均检索覆盖率

## 扩展方向

- 引入异构图表示（实体节点/文档节点）
- 联合训练图嵌入与生成器（端到端）
- 多跳检索与动态图结构学习
- 在图嵌入空间中进行增量学习/冷启动问答

## 引用

如果您在研究中使用了本项目，请引用：

```
@article{graphlightrag2023,
  title={Graph-Enhanced LightRAG: 融合图结构嵌入的结构化检索生成框架研究},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。
