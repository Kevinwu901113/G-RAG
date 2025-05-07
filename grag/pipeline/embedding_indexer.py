#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
向量嵌入与索引构建模块

该模块负责对已处理的文档进行向量嵌入，并构建高效的索引结构。
支持增量更新和批量处理，确保系统能够高效处理大规模文档集合。
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from ..core.base import QueryParam
from ..rag.lightrag import LightRAG
from ..core.llm import ollama_model_complete, ollama_embedding
from ..utils.common import wrap_embedding_func_with_attrs, create_progress_bar
from ..utils.logger_manager import get_logger
from ..core.storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage

# 创建日志器
logger = get_logger("embedding_indexer")

class EmbeddingIndexer:
    """
    向量嵌入与索引构建类，负责文档向量化和索引构建
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化向量嵌入索引器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.working_dir = config["storage"]["working_dir"]
        
        # 确保工作目录存在
        os.makedirs(self.working_dir, exist_ok=True)
        
        # 嵌入配置
        self.embed_config = config["embedding"]
        
        # 初始化RAG系统
        self.rag = None
    
    async def setup_rag(self) -> LightRAG:
        """
        设置并初始化RAG系统
        
        Returns:
            初始化好的LightRAG实例
        """
        # 获取嵌入模型配置
        embed_config = self.config["embedding"]
        
        # 创建嵌入函数
        embedding_func = wrap_embedding_func_with_attrs(
            embedding_dim=embed_config["embedding_dim"],
            max_token_size=embed_config["max_token_size"]
        )(
            lambda texts: ollama_embedding(
                texts, embed_model=embed_config["model_name"], host=embed_config["host"]
            )
        )
        
        # 创建全局配置
        global_config = {
            "working_dir": self.working_dir,
            "embedding_batch_num": self.config["storage"]["embedding_batch_num"],
            "cosine_better_than_threshold": self.config["query"]["cosine_better_than_threshold"],
            "llm_model_name": self.config["llm"]["model_name"],
            "graph_file": os.path.join(self.working_dir, "graph_graph.graphml"),
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
                "host": self.config["llm"]["host"], 
                "options": self.config["llm"]["options"],
                "hashing_kv": doc_full_storage
            },
            embedding_func=embedding_func,
        )
        
        return rag
    
    async def build_indices(self) -> bool:
        """
        构建向量索引
        
        Returns:
            是否成功构建索引
        """
        logger.info("开始构建向量索引")
        
        # 初始化RAG系统
        self.rag = await self.setup_rag()
        
        # 加载已处理的文档列表
        processed_docs_file = os.path.join(self.working_dir, "processed_documents.json")
        if not os.path.exists(processed_docs_file):
            logger.error(f"找不到已处理的文档列表: {processed_docs_file}")
            return False
        
        with open(processed_docs_file, "r", encoding="utf-8") as f:
            processed_docs = json.load(f)
        
        if not processed_docs:
            logger.warning("没有找到已处理的文档")
            return False
        
        # 创建进度条
        total_docs = len(processed_docs)
        progress_bar = create_progress_bar(total_docs, "向量索引构建进度")
        
        # 构建索引
        for i, doc_info in enumerate(processed_docs):
            doc_id = doc_info["doc_id"]
            file_path = doc_info["file_path"]
            file_name = os.path.basename(file_path)
            
            progress_bar.update(i, f"为文档 {file_name} 构建索引")
            
            try:
                # 更新文档的向量嵌入
                await self.rag.update_document_embeddings(doc_id)
                logger.info(f"文档 {file_name} 的向量索引已构建")
            except Exception as e:
                logger.error(f"为文档 {file_name} 构建索引时出错: {str(e)}")
            
            progress_bar.update(i+1, f"完成 {file_name}")
        
        # 完成进度条
        progress_bar.finish()
        
        # 保存索引状态
        index_status_file = os.path.join(self.working_dir, "index_status.json")
        index_status = {
            "total_documents": total_docs,
            "indexed_documents": total_docs,
            "embedding_model": self.embed_config["model_name"],
            "embedding_dim": self.embed_config["embedding_dim"],
            "timestamp": asyncio.get_event_loop().time()
        }
        
        with open(index_status_file, "w", encoding="utf-8") as f:
            json.dump(index_status, f, ensure_ascii=False, indent=2)
        
        logger.info(f"向量索引构建完成，共处理 {total_docs} 个文档，状态已保存到 {index_status_file}")
        
        # 优化索引（可选）
        try:
            await self.optimize_indices()
        except Exception as e:
            logger.warning(f"优化索引时出错: {str(e)}")
        
        return True
    
    async def optimize_indices(self) -> bool:
        """
        优化向量索引，提高查询效率
        
        Returns:
            是否成功优化索引
        """
        logger.info("开始优化向量索引")
        
        try:
            # 优化文档块向量存储
            await self.rag.chunks_vector_storage.optimize()
            
            # 优化实体向量存储
            await self.rag.entity_vector_storage.optimize()
            
            # 优化关系向量存储
            await self.rag.relationship_vector_storage.optimize()
            
            logger.info("向量索引优化完成")
            return True
        except Exception as e:
            logger.error(f"优化向量索引时出错: {str(e)}")
            return False
    
    async def update_indices(self, doc_ids: List[str]) -> bool:
        """
        更新指定文档的向量索引
        
        Args:
            doc_ids: 需要更新的文档ID列表
            
        Returns:
            是否成功更新索引
        """
        logger.info(f"开始更新 {len(doc_ids)} 个文档的向量索引")
        
        # 初始化RAG系统（如果尚未初始化）
        if self.rag is None:
            self.rag = await self.setup_rag()
        
        # 创建进度条
        total_docs = len(doc_ids)
        progress_bar = create_progress_bar(total_docs, "向量索引更新进度")
        
        # 更新索引
        success_count = 0
        for i, doc_id in enumerate(doc_ids):
            progress_bar.update(i, f"更新文档 {doc_id} 的索引")
            
            try:
                # 更新文档的向量嵌入
                await self.rag.update_document_embeddings(doc_id)
                logger.info(f"文档 {doc_id} 的向量索引已更新")
                success_count += 1
            except Exception as e:
                logger.error(f"更新文档 {doc_id} 的索引时出错: {str(e)}")
            
            progress_bar.update(i+1, f"完成 {doc_id}")
        
        # 完成进度条
        progress_bar.finish()
        
        logger.info(f"向量索引更新完成，成功更新 {success_count}/{total_docs} 个文档")
        
        return success_count > 0