#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查询处理器模块

该模块负责处理用户查询，整合各个组件的功能，生成最终回答。
支持多种查询类型和处理策略，提供完整的RAG流程。
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from ..core.base import QueryParam
from ..rag.lightrag import LightRAG
from ..core.llm import ollama_model_complete, ollama_embedding
from ..utils.common import wrap_embedding_func_with_attrs
from ..utils.logger_manager import get_logger
from ..core.storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
from .query_classifier_manager import QueryClassifierManager
from .pruner_manager import PrunerManager

# 创建日志器
logger = get_logger("query_processor")

class QueryProcessor:
    """
    查询处理器类，负责处理用户查询并生成回答
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化查询处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.working_dir = config["storage"]["working_dir"]
        
        # 确保工作目录存在
        os.makedirs(self.working_dir, exist_ok=True)
        
        # 查询配置
        self.query_config = config["query"]
        
        # 初始化RAG系统
        self.rag = None
        
        # 初始化查询分类器
        self.classifier_manager = QueryClassifierManager(config)
        
        # 初始化剪枝管理器
        self.pruner_manager = PrunerManager(config)
    
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
    
    async def process_query(self, query: str) -> Tuple[str, List[str]]:
        """
        处理查询并生成回答
        
        Args:
            query: 查询文本
            
        Returns:
            回答和来源列表
        """
        logger.info(f"开始处理查询: {query}")
        
        # 初始化RAG系统（如果尚未初始化）
        if self.rag is None:
            self.rag = await self.setup_rag()
        
        try:
            # 应用查询分类器
            route_info = await self.classifier_manager.route_query(query)
            query_type = route_info.get("route", "default")
            query_params = route_info.get("params", {})
            
            logger.info(f"查询类型: {query_type}")
            
            # 根据查询类型选择处理策略
            if query_type == "factoid":
                # 事实型查询，使用标准检索+生成
                return await self._process_factoid_query(query, query_params)
            elif query_type == "complex":
                # 复杂查询，使用多跳检索+推理
                return await self._process_complex_query(query, query_params)
            elif query_type == "entity":
                # 实体查询，使用图检索
                return await self._process_entity_query(query, query_params)
            else:
                # 默认处理方式
                return await self._process_default_query(query)
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}")
            return f"处理查询时出错: {str(e)}", []
    
    async def _process_factoid_query(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        处理事实型查询
        
        Args:
            query: 查询文本
            params: 查询参数
            
        Returns:
            回答和来源列表
        """
        logger.info("使用事实型查询处理流程")
        
        # 设置检索参数
        top_k = params.get("top_k", self.query_config.get("factoid_top_k", 5))
        
        # 应用剪枝优化检索结果
        await self.pruner_manager.apply_pruning()
        
        # 检索相关文档
        results = await self.rag.retrieve(query, top_k=top_k)
        
        # 生成回答
        answer, sources = await self.rag.generate_answer(query, results)
        
        return answer, sources
    
    async def _process_complex_query(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        处理复杂查询
        
        Args:
            query: 查询文本
            params: 查询参数
            
        Returns:
            回答和来源列表
        """
        logger.info("使用复杂查询处理流程")
        
        # 设置检索参数
        top_k = params.get("top_k", self.query_config.get("complex_top_k", 8))
        max_hops = params.get("max_hops", self.query_config.get("max_hops", 3))
        
        # 应用剪枝优化检索结果
        await self.pruner_manager.apply_pruning()
        
        # 使用多跳检索
        results = await self.rag.multi_hop_retrieve(query, top_k=top_k, max_hops=max_hops)
        
        # 生成回答
        answer, sources = await self.rag.generate_answer(query, results, use_reasoning=True)
        
        return answer, sources
    
    async def _process_entity_query(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        处理实体查询
        
        Args:
            query: 查询文本
            params: 查询参数
            
        Returns:
            回答和来源列表
        """
        logger.info("使用实体查询处理流程")
        
        # 设置检索参数
        top_k = params.get("top_k", self.query_config.get("entity_top_k", 5))
        
        # 从查询中提取实体
        entities = await self.rag.extract_entities(query)
        
        if not entities:
            logger.warning("未从查询中提取到实体，回退到默认处理流程")
            return await self._process_default_query(query)
        
        # 使用图检索
        results = await self.rag.graph_retrieve(query, entities=entities, top_k=top_k)
        
        # 生成回答
        answer, sources = await self.rag.generate_answer(query, results)
        
        return answer, sources
    
    async def _process_default_query(self, query: str) -> Tuple[str, List[str]]:
        """
        默认查询处理流程
        
        Args:
            query: 查询文本
            
        Returns:
            回答和来源列表
        """
        logger.info("使用默认查询处理流程")
        
        # 设置检索参数
        top_k = self.query_config.get("default_top_k", 5)
        
        # 应用剪枝优化检索结果
        await self.pruner_manager.apply_pruning()
        
        # 检索相关文档
        results = await self.rag.retrieve(query, top_k=top_k)
        
        # 生成回答
        answer, sources = await self.rag.generate_answer(query, results)
        
        return answer, sources
    
    async def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取查询历史记录
        
        Args:
            limit: 返回的记录数量限制
            
        Returns:
            查询历史记录列表
        """
        history_file = os.path.join(self.working_dir, "query_history.json")
        
        if not os.path.exists(history_file):
            return []
        
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            
            # 返回最近的记录
            return history[-limit:]
        except Exception as e:
            logger.error(f"获取查询历史记录时出错: {str(e)}")
            return []
    
    async def save_query_history(self, query: str, answer: str, sources: List[str]) -> bool:
        """
        保存查询历史记录
        
        Args:
            query: 查询文本
            answer: 生成的回答
            sources: 来源列表
            
        Returns:
            是否成功保存
        """
        history_file = os.path.join(self.working_dir, "query_history.json")
        
        try:
            # 加载现有历史记录
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []
            
            # 添加新记录
            history.append({
                "query": query,
                "answer": answer,
                "sources": sources,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # 限制历史记录数量
            max_history = self.config.get("max_query_history", 100)
            if len(history) > max_history:
                history = history[-max_history:]
            
            # 保存历史记录
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"保存查询历史记录时出错: {str(e)}")
            return False