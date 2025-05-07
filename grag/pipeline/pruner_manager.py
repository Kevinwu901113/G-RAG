#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
剪枝模块

该模块负责对检索结果进行剪枝和优化，提高检索质量和效率。
支持多种剪枝策略，并提供训练和评估功能。
"""

import os
import json
import pickle
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union

from ..core.base import QueryParam
from ..rag.lightrag import LightRAG
from ..core.llm import ollama_model_complete, ollama_embedding
from ..utils.common import wrap_embedding_func_with_attrs, create_progress_bar
from ..utils.logger_manager import get_logger
from ..core.storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage

# 创建日志器
logger = get_logger("pruner_manager")

class PrunerManager:
    """
    剪枝管理器类，负责检索结果的剪枝和优化
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化剪枝管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.working_dir = config["storage"]["working_dir"]
        
        # 确保工作目录存在
        os.makedirs(self.working_dir, exist_ok=True)
        
        # 剪枝配置
        self.pruner_config = config.get("pruner", {})
        
        # 模型保存路径
        self.model_dir = os.path.join(self.working_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.model_dir, "pruner_model.pkl")
        
        # 初始化RAG系统
        self.rag = None
        
        # 初始化剪枝模型
        self.pruner_model = None
    
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
    
    def load_model(self) -> bool:
        """
        加载已训练的剪枝模型
        
        Returns:
            是否成功加载模型
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"找不到剪枝模型: {self.model_path}")
            
            # 尝试自动训练模型
            logger.info("尝试自动训练剪枝模型...")
            try:
                import sys
                import subprocess
                from pathlib import Path
                
                # 获取train_pruner.py脚本路径
                script_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "scripts" / "train_pruner.py"
                
                if script_path.exists():
                    # 准备命令行参数
                    config_path = self.config.get("config_path", "config.yaml")
                    
                    # 执行训练脚本
                    cmd = [sys.executable, str(script_path), "--config", config_path, "--output_path", self.model_path]
                    
                    logger.info(f"执行命令: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"已自动训练缺失模型并保存至: {self.model_path}")
                        # 重新尝试加载模型
                        return self.load_model()
                    else:
                        logger.error(f"自动训练剪枝模型失败: {result.stderr}")
                else:
                    logger.error(f"找不到训练脚本: {script_path}")
            except Exception as e:
                logger.error(f"自动训练剪枝模型时出错: {str(e)}")
            
            return False
        
        try:
            with open(self.model_path, "rb") as f:
                self.pruner_model = pickle.load(f)
            
            logger.info(f"成功加载剪枝模型: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"加载剪枝模型时出错: {str(e)}")
            return False
    
    async def train(self, train_data_path: str) -> bool:
        """
        训练剪枝模型
        
        Args:
            train_data_path: 训练数据路径
            
        Returns:
            是否成功训练模型
        """
        if not os.path.exists(train_data_path):
            logger.error(f"找不到训练数据: {train_data_path}")
            return False
        
        logger.info(f"开始训练剪枝模型，使用数据: {train_data_path}")
        
        try:
            # 加载训练数据
            with open(train_data_path, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            
            if not train_data or not isinstance(train_data, list):
                logger.error("训练数据格式不正确，应为JSON数组")
                return False
            
            # 初始化RAG系统（如果尚未初始化）
            if self.rag is None:
                self.rag = await self.setup_rag()
            
            # 提取训练特征和标签
            features = []
            labels = []
            
            for item in train_data:
                query = item.get("query", "")
                relevant_chunks = item.get("relevant_chunks", [])
                irrelevant_chunks = item.get("irrelevant_chunks", [])
                
                if not query or not relevant_chunks:
                    continue
                
                # 获取查询的向量表示
                query_embedding = await self.rag.embedding_func([query])
                
                # 为相关块创建正样本
                for chunk_id in relevant_chunks:
                    try:
                        chunk_data = await self.rag.doc_chunks_storage.get(chunk_id)
                        if not chunk_data:
                            continue
                            
                        chunk_text = chunk_data.get("text", "")
                        chunk_embedding = await self.rag.embedding_func([chunk_text])
                        
                        # 计算特征（如相似度、文本长度等）
                        feature = self._extract_features(query, query_embedding[0], chunk_text, chunk_embedding[0])
                        features.append(feature)
                        labels.append(1)  # 相关
                    except Exception as e:
                        logger.warning(f"处理相关块 {chunk_id} 时出错: {str(e)}")
                
                # 为不相关块创建负样本
                for chunk_id in irrelevant_chunks:
                    try:
                        chunk_data = await self.rag.doc_chunks_storage.get(chunk_id)
                        if not chunk_data:
                            continue
                            
                        chunk_text = chunk_data.get("text", "")
                        chunk_embedding = await self.rag.embedding_func([chunk_text])
                        
                        # 计算特征
                        feature = self._extract_features(query, query_embedding[0], chunk_text, chunk_embedding[0])
                        features.append(feature)
                        labels.append(0)  # 不相关
                    except Exception as e:
                        logger.warning(f"处理不相关块 {chunk_id} 时出错: {str(e)}")
            
            if not features or not labels:
                logger.error("无法从训练数据中提取有效特征和标签")
                return False
            
            # 训练模型（使用随机森林或其他分类器）
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # 分割训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            # 创建并训练模型
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # 评估模型
            val_accuracy = model.score(X_val, y_val)
            logger.info(f"剪枝模型验证准确率: {val_accuracy:.4f}")
            
            # 保存模型
            self.pruner_model = model
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)
            
            logger.info(f"剪枝模型训练完成，已保存到: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"训练剪枝模型时出错: {str(e)}")
            return False
    
    def _extract_features(self, query: str, query_embedding: List[float], 
                          chunk_text: str, chunk_embedding: List[float]) -> List[float]:
        """
        从查询和文档块中提取特征
        
        Args:
            query: 查询文本
            query_embedding: 查询的向量表示
            chunk_text: 文档块文本
            chunk_embedding: 文档块的向量表示
            
        Returns:
            特征向量
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 计算余弦相似度
        cosine_sim = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(chunk_embedding).reshape(1, -1)
        )[0][0]
        
        # 文本长度特征
        query_length = len(query)
        chunk_length = len(chunk_text)
        length_ratio = chunk_length / max(query_length, 1)
        
        # 词汇重叠特征
        query_words = set(query.lower().split())
        chunk_words = set(chunk_text.lower().split())
        word_overlap = len(query_words.intersection(chunk_words))
        word_overlap_ratio = word_overlap / max(len(query_words), 1)
        
        # 返回特征向量
        return [cosine_sim, length_ratio, word_overlap, word_overlap_ratio]
    
    async def apply_pruning(self) -> bool:
        """
        应用剪枝模型优化检索结果
        
        Returns:
            是否成功应用剪枝
        """
        logger.info("开始应用剪枝模型优化检索结果")
        
        # 加载模型（如果尚未加载）
        if self.pruner_model is None:
            if not self.load_model():
                logger.error("无法加载剪枝模型")
                return False
        
        # 初始化RAG系统（如果尚未初始化）
        if self.rag is None:
            self.rag = await self.setup_rag()
        
        # 应用剪枝策略到RAG系统
        try:
            # 设置自定义检索过滤器
            self.rag.set_retrieval_filter(self.filter_retrieval_results)
            
            logger.info("剪枝模型已应用到检索系统")
            return True
        except Exception as e:
            logger.error(f"应用剪枝模型时出错: {str(e)}")
            return False
    
    async def filter_retrieval_results(self, query: str, results: List[Dict[str, Any]], 
                                     top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用剪枝模型过滤检索结果
        
        Args:
            query: 查询文本
            results: 原始检索结果列表
            top_k: 返回的结果数量
            
        Returns:
            过滤后的检索结果列表
        """
        if not results:
            return results
        
        if self.pruner_model is None:
            logger.warning("剪枝模型未加载，返回原始结果")
            return results[:top_k]
        
        try:
            # 获取查询的向量表示
            query_embedding = await self.rag.embedding_func([query])
            
            # 为每个结果计算特征和预测相关性分数
            scored_results = []
            
            for result in results:
                chunk_text = result.get("text", "")
                if not chunk_text:
                    continue
                    
                # 获取文档块的向量表示
                chunk_embedding = await self.rag.embedding_func([chunk_text])
                
                # 提取特征
                feature = self._extract_features(query, query_embedding[0], chunk_text, chunk_embedding[0])
                
                # 预测相关性概率
                relevance_prob = self.pruner_model.predict_proba([feature])[0][1]  # 类别1的概率
                
                # 添加到结果列表
                scored_result = result.copy()
                scored_result["relevance_score"] = float(relevance_prob)
                scored_results.append(scored_result)
            
            # 按相关性分数排序
            sorted_results = sorted(scored_results, key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # 应用阈值过滤（可选）
            threshold = self.pruner_config.get("relevance_threshold", 0.5)
            filtered_results = [r for r in sorted_results if r.get("relevance_score", 0) >= threshold]
            
            # 返回前top_k个结果
            return filtered_results[:top_k]
        except Exception as e:
            logger.error(f"过滤检索结果时出错: {str(e)}")
            return results[:top_k]  # 出错时返回原始结果
    
    async def evaluate(self, eval_data_path: str) -> Dict[str, float]:
        """
        评估剪枝模型性能
        
        Args:
            eval_data_path: 评估数据路径
            
        Returns:
            评估指标字典
        """
        if not os.path.exists(eval_data_path):
            logger.error(f"找不到评估数据: {eval_data_path}")
            return {}
        
        if self.pruner_model is None:
            if not self.load_model():
                logger.error("无法加载剪枝模型")
                return {}
        
        logger.info(f"开始评估剪枝模型，使用数据: {eval_data_path}")
        
        try:
            # 加载评估数据
            with open(eval_data_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            
            if not eval_data or not isinstance(eval_data, list):
                logger.error("评估数据格式不正确，应为JSON数组")
                return {}
            
            # 初始化RAG系统（如果尚未初始化）
            if self.rag is None:
                self.rag = await self.setup_rag()
            
            # 评估指标
            total_queries = 0
            precision_sum = 0
            recall_sum = 0
            f1_sum = 0
            ndcg_sum = 0
            
            # 处理每个查询
            for item in eval_data:
                query = item.get("query", "")
                relevant_chunks = set(item.get("relevant_chunks", []))
                
                if not query or not relevant_chunks:
                    continue
                
                total_queries += 1
                
                # 获取原始检索结果
                raw_results = await self.rag.retrieve(query, top_k=20)
                raw_chunk_ids = [r.get("id") for r in raw_results if "id" in r]
                
                # 应用剪枝过滤
                filtered_results = await self.filter_retrieval_results(query, raw_results, top_k=10)
                filtered_chunk_ids = [r.get("id") for r in filtered_results if "id" in r]
                
                # 计算精确率和召回率
                retrieved_relevant = set(filtered_chunk_ids).intersection(relevant_chunks)
                precision = len(retrieved_relevant) / max(len(filtered_chunk_ids), 1)
                recall = len(retrieved_relevant) / max(len(relevant_chunks), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-10)
                
                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                
                # 计算NDCG
                ndcg = self._calculate_ndcg(filtered_chunk_ids, relevant_chunks)
                ndcg_sum += ndcg
            
            if total_queries == 0:
                logger.error("没有有效的评估查询")
                return {}
            
            # 计算平均指标
            avg_precision = precision_sum / total_queries
            avg_recall = recall_sum / total_queries
            avg_f1 = f1_sum / total_queries
            avg_ndcg = ndcg_sum / total_queries
            
            # 汇总评估结果
            metrics = {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "ndcg": avg_ndcg,
                "queries": total_queries
            }
            
            # 保存评估结果
            eval_result_path = os.path.join(self.model_dir, "pruner_eval.json")
            with open(eval_result_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            logger.info(f"剪枝模型评估完成，F1分数: {avg_f1:.4f}，结果已保存到: {eval_result_path}")
            
            return metrics
        except Exception as e:
            logger.error(f"评估剪枝模型时出错: {str(e)}")
            return {}
    
    def _calculate_ndcg(self, results: List[str], relevant_docs: set, k: int = 10) -> float:
        """
        计算NDCG指标
        
        Args:
            results: 检索结果ID列表
            relevant_docs: 相关文档ID集合
            k: 计算NDCG@k
            
        Returns:
            NDCG值
        """
        import numpy as np
        
        # 限制结果数量
        results = results[:k]
        
        # 创建相关性列表（1表示相关，0表示不相关）
        relevance = [1 if doc_id in relevant_docs else 0 for doc_id in results]
        
        # 计算DCG
        dcg = 0
        for i, rel in enumerate(relevance):
            dcg += rel / np.log2(i + 2)  # i+2 是因为log2(1)=0
        
        # 计算理想DCG（将所有相关文档排在前面）
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = 0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)
        
        # 计算NDCG
        ndcg = dcg / max(idcg, 1e-10)
        
        return ndcg