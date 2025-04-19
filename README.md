# GRAG: Graph-based Retrieval-Augmented Generation

GRAG是一个基于图的检索增强生成系统，它结合了知识图谱和向量检索的优势，为大语言模型提供更丰富的上下文信息。

## 特性

- **多模式检索**：支持本地(local)、全局(global)、混合(hybrid)和朴素(naive)四种检索模式
- **知识图谱集成**：自动从文档中提取实体和关系，构建知识图谱
- **灵活的存储后端**：支持多种存储后端，包括本地文件、Neo4j和Oracle
- **多种嵌入模型**：支持OpenAI、Azure OpenAI、Amazon Bedrock、Ollama和Hugging Face等多种嵌入模型
- **异步API**：全异步API设计，支持高并发查询
- **流式响应**：支持流式生成回答，提供更好的用户体验

## 安装

```bash
# 基本安装
pip install -e .

# 安装所有可选依赖
pip install -e ".[all]"

# 安装特定可选依赖
pip install -e ".[neo4j,docx]"
```

## 快速开始

### 基本用法

```python
import asyncio
from grag.core.base import QueryParam
from grag.rag.lightrag import LightRAG
from grag.core.llm import ollama_model_complete, ollama_embedding
from grag.utils.common import wrap_embedding_func_with_attrs
from grag.core.storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage

async def main():
    # 创建嵌入函数
    embedding_func = wrap_embedding_func_with_attrs(
        embedding_dim=1024,
        max_token_size=8192
    )(
        lambda texts: ollama_embedding(
            texts, embed_model="bge-m3", host="http://localhost:11434"
        )
    )
    
    # 创建全局配置
    global_config = {
        "working_dir": "./result/grag_data",
        "embedding_batch_num": 32,
        "cosine_better_than_threshold": 0.2,
        "llm_model_name": "qwen2.5:7b-instruct-fp16",
    }
    
    # 创建存储实例
    doc_full_storage = JsonKVStorage(
        namespace="full_docs",
        global_config=global_config,
        embedding_func=embedding_func
    )
    
    doc_chunks_storage = JsonKVStorage(
        namespace="text_chunks",
        global_config=global_config,
        embedding_func=embedding_func
    )
    
    chunks_vector_storage = NanoVectorDBStorage(
        namespace="chunks",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"tokens", "chunk_order_index", "full_doc_id"}
    )
    
    entity_vector_storage = NanoVectorDBStorage(
        namespace="entities",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"entity_name"}
    )
    
    relationship_vector_storage = NanoVectorDBStorage(
        namespace="relationships",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"src_id", "tgt_id"}
    )
    
    graph_storage = NetworkXStorage(
        namespace="graph",
        global_config=global_config,
        embedding_func=embedding_func
    )
    
    # 创建LightRAG实例
    rag = LightRAG(
        doc_full_storage=doc_full_storage,
        doc_chunks_storage=doc_chunks_storage,
        chunks_vector_storage=chunks_vector_storage,
        entity_vector_storage=entity_vector_storage,
        relationship_vector_storage=relationship_vector_storage,
        graph_storage=graph_storage,
        llm_model_func=ollama_model_complete,
        llm_model_kwargs={
            "host": "http://localhost:11434", 
            "options": {"num_ctx": 32768}
        },
        embedding_func=embedding_func,
    )
    
    # 索引文档
    doc_id = await rag.index_document("这是一个示例文档，用于测试GRAG系统。")
    
    # 查询
    answer = await rag.aquery("这个文档是关于什么的？", param=QueryParam(mode="hybrid"))
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
```

### 使用不同的检索模式

GRAG支持四种检索模式：

1. **本地模式(local)**：基于实体的检索，适合查询特定实体的信息
2. **全局模式(global)**：基于关系的检索，适合查询实体间关系的信息
3. **混合模式(hybrid)**：结合本地和全局模式，提供更全面的上下文
4. **朴素模式(naive)**：直接基于文本块的检索，适合简单查询

```python
from grag.core.base import QueryParam

# 本地模式
answer_local = await rag.aquery("实体A的信息是什么？", param=QueryParam(mode="local"))

# 全局模式
answer_global = await rag.aquery("实体A和实体B之间的关系是什么？", param=QueryParam(mode="global"))

# 混合模式
answer_hybrid = await rag.aquery("实体A的信息及其与其他实体的关系是什么？", param=QueryParam(mode="hybrid"))

# 朴素模式
answer_naive = await rag.aquery("文档中提到了什么内容？", param=QueryParam(mode="naive"))
```

### 流式响应

GRAG支持流式生成回答，提供更好的用户体验：

```python
async def stream_example():
    # 流式生成回答
    async for token in rag.astream("这个文档是关于什么的？", param=QueryParam(mode="hybrid")):
        print(token, end="", flush=True)
    print()

asyncio.run(stream_example())
```

## 项目结构

```
grag/
├── grag/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py          # 基础类和接口定义
│   │   ├── llm.py           # LLM集成
│   │   ├── prompt.py        # 提示模板
│   │   ├── storage.py       # 存储实现
│   │   └── stream.py        # 流式响应支持
│   ├── kg/
│   │   ├── __init__.py
│   │   ├── neo4j_impl.py    # Neo4j存储实现
│   │   └── oracle_impl.py   # Oracle存储实现
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── lightrag.py      # 核心RAG实现
│   │   ├── query.py         # 查询功能
│   │   └── embedding.py     # 嵌入功能
│   └── utils/
│       ├── __init__.py
│       └── common.py        # 通用工具函数
├── main.py                  # 主程序入口
├── setup.py                 # 安装脚本
└── README.md                # 项目说明
```

## 许可证

MIT
