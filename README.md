# GRAG: 基于图的检索增强生成系统

GRAG是一个高级检索增强生成(RAG)系统，它结合了知识图谱和向量检索的优势，为大语言模型提供更丰富、更精确的上下文信息。通过创新的流水线架构，GRAG能够处理复杂查询，提供高质量的回答。

## 系统架构

GRAG采用模块化流水线架构，包含四个核心模块：

1. **文档处理器**：负责扫描数据目录，读取文档，进行预处理，并构建图结构
2. **向量嵌入与索引**：对处理后的文档进行向量嵌入，构建高效索引结构
3. **查询分类器**：自动分析查询类型，选择最佳检索策略
4. **剪枝模块**：优化检索结果，提高回答质量和效率

![GRAG系统架构](https://example.com/grag_architecture.png)

## 核心特性

- **智能查询路由**：自动识别查询类型（事实型、复杂型、实体型），选择最佳处理策略
- **多模式检索**：支持本地(local)、全局(global)、混合(hybrid)和朴素(naive)四种检索模式
- **知识图谱集成**：自动从文档中提取实体和关系，构建知识图谱
- **灵活的存储后端**：支持多种存储后端，包括本地文件、Neo4j和Oracle
- **多种嵌入模型**：支持OpenAI、Azure OpenAI、Amazon Bedrock、Ollama和Hugging Face等多种嵌入模型
- **异步API**：全异步API设计，支持高并发查询
- **流式响应**：支持流式生成回答，提供更好的用户体验
- **自适应剪枝**：智能优化检索结果，提高回答质量

## 安装

```bash
# 基本安装
pip install -e .

# 安装所有可选依赖
pip install -e ".[all]"

# 安装特定可选依赖
pip install -e ".[neo4j,docx]"
```

## 配置

GRAG使用YAML格式的配置文件，包含以下主要部分：

```yaml
# LLM模型配置
llm:
  provider: "ollama"  # 可选: "ollama", "openai", "azure_openai", "bedrock", "huggingface"
  model_name: "qwen2.5:7b-instruct-fp16"
  host: "http://localhost:11434"

# 嵌入模型配置
embedding:
  provider: "ollama"  
  model_name: "bge-m3"
  embedding_dim: 1024
  max_token_size: 8192

# 文档处理配置
document:
  chunk_size: 1000
  chunk_overlap: 200

# 存储配置
storage:
  working_dir: "./result/grag_data"
```

完整配置示例请参考项目中的`config.yaml`文件。

## 使用方法

### 流水线模式

GRAG提供了完整的流水线模式，可以一次性执行所有步骤，也可以单独执行特定步骤：

```bash
# 执行完整流水线
python main_pipeline.py --config config.yaml --data_dir ./data --all

# 只执行文档处理步骤
python main_pipeline.py --config config.yaml --data_dir ./data --document_processing

# 只执行向量嵌入步骤
python main_pipeline.py --config config.yaml --embedding_indexing

# 执行查询
python main_pipeline.py --config config.yaml --query "这是一个测试查询"
```

### 编程接口

```python
import asyncio
from grag.pipeline.query_processor import QueryProcessor
import yaml

async def main():
    # 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 创建查询处理器
    processor = QueryProcessor(config)
    
    # 处理查询
    answer, sources = await processor.process_query("这是一个测试查询")
    print(f"回答: {answer}")
    print(f"来源: {sources}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 核心模块详解

### 1. 文档处理器

文档处理器负责扫描数据目录，读取文档，进行预处理，并构建图结构。支持多种文档格式，包括TXT、DOCX和JSON（HotpotQA数据集）。

```python
from grag.pipeline.document_processor import DocumentProcessor

# 创建文档处理器
processor = DocumentProcessor(config)

# 处理单个文件
doc_id = await processor.process_file("path/to/document.txt")

# 处理整个目录
success = await processor.process_directory("path/to/data_dir")
```

### 2. 向量嵌入与索引

向量嵌入模块负责对处理后的文档进行向量嵌入，构建高效索引结构。支持增量更新和批量处理。

```python
from grag.pipeline.embedding_indexer import EmbeddingIndexer

# 创建嵌入索引器
indexer = EmbeddingIndexer(config)

# 构建索引
success = await indexer.build_indices()

# 更新索引
success = await indexer.update_indices()
```

### 3. 查询分类器

查询分类器负责自动分析查询类型，选择最佳检索策略。支持事实型、复杂型和实体型查询。

```python
from grag.pipeline.query_classifier_manager import QueryClassifierManager

# 创建查询分类器管理器
classifier = QueryClassifierManager(config)

# 路由查询
route_info = await classifier.route_query("这是一个测试查询")
print(f"查询类型: {route_info['route']}")
print(f"查询参数: {route_info['params']}")
```

### 4. 剪枝模块

剪枝模块负责优化检索结果，提高回答质量和效率。支持多种剪枝策略，并提供训练和评估功能。

```python
from grag.pipeline.pruner_manager import PrunerManager

# 创建剪枝管理器
pruner = PrunerManager(config)

# 应用剪枝
await pruner.apply_pruning()

# 训练剪枝模型
success = await pruner.train("path/to/train_data.json")

# 评估剪枝效果
metrics = await pruner.evaluate("path/to/test_data.json")
```

## 高级用法

### 使用不同的检索模式

GRAG支持四种检索模式：

1. **本地模式(local)**：基于实体的检索，适合查询特定实体的信息
2. **全局模式(global)**：基于关系的检索，适合查询实体间关系的信息
3. **混合模式(hybrid)**：结合本地和全局模式，提供更全面的上下文
4. **朴素模式(naive)**：直接基于文本块的检索，适合简单查询

```python
from grag.core.base import QueryParam

# 本地模式
answer_local, sources = await processor.process_query(
    "实体A的信息是什么？", 
    {"route": "entity", "params": {"mode": "local"}}
)

# 全局模式
answer_global, sources = await processor.process_query(
    "实体A和实体B之间的关系是什么？", 
    {"route": "complex", "params": {"mode": "global"}}
)

# 混合模式
answer_hybrid, sources = await processor.process_query(
    "实体A的信息及其与其他实体的关系是什么？", 
    {"route": "complex", "params": {"mode": "hybrid"}}
)

# 朴素模式
answer_naive, sources = await processor.process_query(
    "文档中提到了什么内容？", 
    {"route": "factoid", "params": {"mode": "naive"}}
)
```

### 流式响应

GRAG支持流式生成回答，提供更好的用户体验：

```python
async def stream_example():
    processor = QueryProcessor(config)
    
    # 流式生成回答
    async for token in processor.stream_query("这个文档是关于什么的？"):
        print(token, end="", flush=True)
    print()

asyncio.run(stream_example())
```

## 项目结构

```
grag/
├── grag/
│   ├── __init__.py
│   ├── core/             # 核心组件
│   │   ├── base.py       # 基础类和接口
│   │   ├── llm.py        # LLM集成
│   │   ├── prompt.py     # 提示模板
│   │   ├── storage.py    # 存储实现
│   │   └── stream.py     # 流式响应
│   ├── kg/               # 知识图谱组件
│   │   ├── neo4j_impl.py # Neo4j实现
│   │   └── oracle_impl.py# Oracle实现
│   ├── pipeline/         # 流水线组件
│   │   ├── document_processor.py    # 文档处理
│   │   ├── embedding_indexer.py     # 向量嵌入
│   │   ├── query_classifier_manager.py # 查询分类
│   │   ├── pruner_manager.py        # 剪枝优化
│   │   └── query_processor.py       # 查询处理
│   ├── rag/              # RAG核心实现
│   │   ├── lightrag.py   # 轻量级RAG
│   │   ├── query.py      # 查询功能
│   │   └── embedding.py  # 嵌入功能
│   └── utils/            # 工具函数
│       ├── common.py     # 通用工具
│       ├── file_scanner.py # 文件扫描
│       └── logger_manager.py # 日志管理
├── main_pipeline.py      # 主流水线入口
├── config.yaml           # 配置文件
└── README.md             # 项目说明
```

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议。请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

MIT
