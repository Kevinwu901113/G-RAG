#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文档预处理与图结构构建模块

该模块负责扫描数据目录，读取文档，进行预处理，并构建图结构。
处理结果将保存为结构化图或JSON文件，作为后续步骤的输入。
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
from ..utils.file_scanner import scan_data_directory
from ..rag.hotpotqa_processor import process_hotpotqa_dataset

# 创建日志器
logger = get_logger("document_processor")

class DocumentProcessor:
    """
    文档处理器类，负责文档预处理与图结构构建
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文档处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.working_dir = config["storage"]["working_dir"]
        
        # 确保工作目录存在
        os.makedirs(self.working_dir, exist_ok=True)
        
        # 文档处理配置
        self.doc_config = config["document"]
        
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
    
    async def process_file(self, file_path: str) -> Optional[str]:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档ID，如果处理失败则返回None
        """
        file_name = os.path.basename(file_path)
        logger.info(f"正在处理文档: {file_path}")
        
        try:
            # 根据文件类型读取内容
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 索引文档
                doc_id = await self.rag.index_document(
                    content, 
                    chunk_size=self.doc_config["chunk_size"], 
                    chunk_overlap=self.doc_config["chunk_overlap"]
                )
                logger.info(f"文档已索引，ID: {doc_id}")
                return doc_id
                
            elif file_path.endswith('.docx'):
                # 使用unstructured库处理docx文件
                try:
                    from unstructured.partition.docx import partition_docx
                    elements = partition_docx(file_path)
                    content = "\n".join([str(e) for e in elements])
                    # 索引文档
                    doc_id = await self.rag.index_document(
                        content, 
                        chunk_size=self.doc_config["chunk_size"], 
                        chunk_overlap=self.doc_config["chunk_overlap"]
                    )
                    logger.info(f"文档已索引，ID: {doc_id}")
                    return doc_id
                except ImportError:
                    logger.error("请安装unstructured库以处理docx文件: pip install unstructured")
                    return None
                    
            elif file_path.endswith('.json'):
                # 处理HotpotQA数据集
                try:
                    doc_id, _ = await process_hotpotqa_dataset(self.rag, file_path)
                    if doc_id:
                        logger.info(f"HotpotQA数据集已索引，ID: {doc_id}")
                        return doc_id
                    else:
                        logger.warning(f"处理HotpotQA数据集失败: {file_path}")
                        return None
                except Exception as e:
                    logger.error(f"处理JSON文件时出错: {e}")
                    return None
            else:
                logger.warning(f"不支持的文件类型: {file_path}")
                return None
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            return None
    
    async def process_directory(self, data_dir: str) -> bool:
        """
        处理数据目录中的所有文件
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            是否成功完成处理
        """
        logger.info(f"开始扫描数据目录: {data_dir}")
        
        # 初始化RAG系统
        self.rag = await self.setup_rag()
        
        # 扫描数据目录
        file_types = scan_data_directory(data_dir)
        
        # 合并所有文件路径
        all_files = []
        for file_type, files in file_types.items():
            if file_type != "other":  # 跳过不支持的文件类型
                all_files.extend(files)
        
        if not all_files:
            logger.warning(f"数据目录 {data_dir} 中没有找到支持的文件")
            return False
        
        # 创建进度条
        total_files = len(all_files)
        progress_bar = create_progress_bar(total_files, "文档处理进度")
        
        # 存储处理结果
        processed_docs = []
        
        # 处理所有文件
        for i, file_path in enumerate(all_files):
            file_name = os.path.basename(file_path)
            progress_bar.update(i, f"处理 {file_name}")
            
            doc_id = await self.process_file(file_path)
            if doc_id:
                processed_docs.append({
                    "file_path": file_path,
                    "doc_id": doc_id
                })
            
            progress_bar.update(i+1, f"完成 {file_name}")
        
        # 完成进度条
        progress_bar.finish()
        
        # 保存处理结果
        result_file = os.path.join(self.working_dir, "processed_documents.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(processed_docs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"文档处理完成，共处理 {len(processed_docs)} 个文件，结果保存到 {result_file}")
        
        # 保存图结构
        graph_file = os.path.join(self.working_dir, "graph_structure.json")
        try:
            graph_data = await self.rag.graph_storage.export_graph_data()
            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            logger.info(f"图结构已保存到 {graph_file}")
        except Exception as e:
            logger.error(f"保存图结构时出错: {str(e)}")
        
        return len(processed_docs) > 0