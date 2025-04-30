#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查询分类器集成模块

该模块提供了将查询分类器集成到RAG系统的功能，用于自动决策查询策略和精度需求。
"""

import os
import torch
from typing import Dict, Optional, Tuple, Any

from .query_classifier import QueryClassifier
from ..utils.logger_manager import get_logger

# 创建日志器
logger = get_logger("query_classifier_integration")

class QueryClassifierIntegration:
    """
    查询分类器集成类，用于在RAG系统中自动决策查询策略和精度需求
    """
    def __init__(self, model_path: str = "./query_classifier_model", default_strategy: str = "hybrid", default_precision: str = "yes"):
        """
        初始化查询分类器集成
        
        Args:
            model_path: 模型路径
            default_strategy: 默认查询策略，当模型不可用时使用
            default_precision: 默认精度需求，当模型不可用时使用
        """
        self.model_path = model_path
        self.default_strategy = default_strategy
        self.default_precision = default_precision
        self.classifier = None
        
        # 尝试加载模型
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        加载查询分类器模型
        
        Returns:
            是否成功加载模型
        """
        try:
            if os.path.exists(self.model_path):
                logger.info(f"从 {self.model_path} 加载查询分类器模型")
                self.classifier = QueryClassifier(self.model_path)
                return True
            else:
                logger.warning(f"模型路径 {self.model_path} 不存在，将使用默认策略")
                return False
        except Exception as e:
            logger.error(f"加载查询分类器模型时出错: {str(e)}")
            return False
    
    def classify_query(self, query: str) -> Dict[str, str]:
        """
        对查询进行分类，决定查询策略和精度需求
        
        Args:
            query: 用户查询文本
            
        Returns:
            包含查询策略和精度需求的字典
        """
        # 如果模型不可用，返回默认值
        if self.classifier is None:
            if not self._load_model():
                logger.warning(f"查询分类器不可用，使用默认策略: {self.default_strategy}, {self.default_precision}")
                return {
                    "query_strategy": self.default_strategy,
                    "precision_required": self.default_precision
                }
        
        try:
            # 使用模型进行预测
            result = self.classifier.predict(query)
            logger.info(f"查询分类结果: {result}")
            return result
        except Exception as e:
            logger.error(f"查询分类过程中出错: {str(e)}")
            # 出错时返回默认值
            return {
                "query_strategy": self.default_strategy,
                "precision_required": self.default_precision
            }
    
    def get_rag_config(self, query: str) -> Dict[str, Any]:
        """
        根据查询获取RAG配置
        
        Args:
            query: 用户查询文本
            
        Returns:
            RAG配置字典
        """
        # 对查询进行分类
        classification = self.classify_query(query)
        
        # 根据分类结果生成RAG配置
        config = {
            "use_rag": classification["query_strategy"] != "noRAG",
            "retrieval_strategy": "hybrid" if classification["query_strategy"] == "hybrid" else "none",
            "high_precision": classification["precision_required"] == "yes",
            # 可以根据需要添加更多配置项
        }
        
        logger.info(f"生成RAG配置: {config}")
        return config

# 创建默认实例，方便直接导入使用
default_classifier = QueryClassifierIntegration()

def get_rag_config(query: str) -> Dict[str, Any]:
    """
    便捷函数，根据查询获取RAG配置
    
    Args:
        query: 用户查询文本
        
    Returns:
        RAG配置字典
    """
    return default_classifier.get_rag_config(query)

def classify_query(query: str) -> Dict[str, str]:
    """
    便捷函数，对查询进行分类
    
    Args:
        query: 用户查询文本
        
    Returns:
        包含查询策略和精度需求的字典
    """
    return default_classifier.classify_query(query)