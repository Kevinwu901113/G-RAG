"""
GRAG: Graph-based Retrieval-Augmented Generation
"""

__version__ = "0.1.0"

from grag.core.base import QueryParam
from grag.rag.lightrag import LightRAG

# 方便导入的别名
from grag.rag.query import query_with_mode, batch_query, compare_query_modes, evaluate_query
from grag.rag.embedding import (
    EmbeddingManager,
    create_openai_embedding_func,
    create_azure_openai_embedding_func,
    create_bedrock_embedding_func,
    create_ollama_embedding_func,
    create_huggingface_embedding_func,
)

# 导出主要类和函数
__all__ = [
    "LightRAG",
    "QueryParam",
    "query_with_mode",
    "batch_query",
    "compare_query_modes",
    "evaluate_query",
    "EmbeddingManager",
    "create_openai_embedding_func",
    "create_azure_openai_embedding_func",
    "create_bedrock_embedding_func",
    "create_ollama_embedding_func",
    "create_huggingface_embedding_func",
]
