#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查询分类器集成模块

该模块负责集成和管理查询分类器，对输入查询进行分类，并路由到合适的处理流程。
支持多种分类器模型，并提供训练和评估功能。
"""

import os
import json
import pickle
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union

from ..rag.query_classifier import QueryClassifier, LABEL_MAPS
from ..rag.query_classifier_integration import QueryClassifierIntegration
from ..utils.logger_manager import get_logger
from ..utils.common import create_progress_bar

# 创建日志器
logger = get_logger("query_classifier_manager")

class QueryClassifierManager:
    """
    查询分类器管理器类，负责查询分类器的训练、评估和应用
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化查询分类器管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.working_dir = config["storage"]["working_dir"]
        
        # 确保工作目录存在
        os.makedirs(self.working_dir, exist_ok=True)
        
        # 分类器配置
        self.classifier_config = config.get("query_classifier", {})
        
        # 模型保存路径
        self.model_dir = os.path.join(self.working_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.model_dir, "query_classifier.pkl")
        
        # 初始化分类器
        self.classifier = None
        self.classifier_integration = None
    
    def load_model(self) -> bool:
        """
        加载已训练的分类器模型
        
        Returns:
            是否成功加载模型
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"找不到分类器模型: {self.model_path}")
            
            # 尝试自动训练模型
            logger.info("尝试自动训练查询分类器模型...")
            try:
                from ..classifier.continual_trainer import ContinualTrainer
                
                # 创建训练器实例
                trainer = ContinualTrainer(
                    config_path=None,  # 使用默认配置
                    model_path=self.model_path,
                    backup_model_dir=os.path.join(self.model_dir, "backups")
                )
                
                # 获取默认训练数据路径
                default_train_data = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "train_data", "default_query_samples.json"
                )
                
                # 如果默认训练数据不存在，尝试使用配置中指定的路径
                if not os.path.exists(default_train_data):
                    default_train_data = self.classifier_config.get("default_train_data", "")
                
                if os.path.exists(default_train_data):
                    # 获取模型配置
                    local_model_path = self.classifier_config.get("local_model_path", None)
                    model_name = self.classifier_config.get("model_name", "bert-base-uncased")
                    
                    # 设置训练器配置
                    trainer.config = {
                        "local_model_path": local_model_path,
                        "model_name": model_name
                    }
                    
                    # 执行训练
                    success = trainer.train(default_train_data)
                    if success:
                        logger.info(f"已自动训练缺失模型并保存至: {self.model_path}")
                        # 重新尝试加载模型
                        return self.load_model()
                    else:
                        logger.error("自动训练查询分类器模型失败")
                else:
                    logger.error(f"找不到默认训练数据: {default_train_data}")
            except Exception as e:
                logger.error(f"自动训练查询分类器模型时出错: {str(e)}")
            
            return False
        
        try:
            with open(self.model_path, "rb") as f:
                self.classifier = pickle.load(f)
            
            # 创建分类器集成
            self.classifier_integration = QueryClassifierIntegration(
                classifier=self.classifier,
                config=self.config
            )
            
            logger.info(f"成功加载分类器模型: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"加载分类器模型时出错: {str(e)}")
            return False
    
    async def train(self, train_data_path: str) -> bool:
        """
        训练查询分类器
        
        Args:
            train_data_path: 训练数据路径
            
        Returns:
            是否成功训练模型
        """
        if not os.path.exists(train_data_path):
            logger.error(f"找不到训练数据: {train_data_path}")
            return False
        
        logger.info(f"开始训练查询分类器，使用数据: {train_data_path}")
        
        try:
            # 加载训练数据
            with open(train_data_path, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            
            if not train_data or not isinstance(train_data, list):
                logger.error("训练数据格式不正确，应为JSON数组")
                return False
            
            # 提取查询和标签
            queries = [item.get("query", "") for item in train_data if "query" in item]
            labels = [item.get("label", "") for item in train_data if "label" in item]
            
            if len(queries) != len(labels) or len(queries) == 0:
                logger.error(f"训练数据不完整: {len(queries)} 个查询, {len(labels)} 个标签")
                return False
            
            # 使用train_model函数直接训练模型
            from ..rag.query_classifier import train_model
            
            # 将训练数据保存为临时文件
            temp_data_path = os.path.join(os.path.dirname(self.model_path), "temp_train_data.jsonl")
            os.makedirs(os.path.dirname(temp_data_path), exist_ok=True)
            
            with open(temp_data_path, "w", encoding="utf-8") as f:
                for query, label in zip(queries, labels):
                    # 解析标签，假设label格式为"query_strategy:precision_required"
                    parts = label.split(":")
                    query_strategy = parts[0] if len(parts) > 0 else "noRAG"
                    precision_required = parts[1] if len(parts) > 1 else "no"
                    
                    # 确保标签值有效
                    if query_strategy not in LABEL_MAPS["query_strategy"]:
                        query_strategy = "noRAG"
                    if precision_required not in LABEL_MAPS["precision_required"]:
                        precision_required = "no"
                    
                    f.write(json.dumps({"query": query, "query_strategy": query_strategy, "precision_required": precision_required}) + "\n")
            
            # 训练模型
            logger.info(f"开始训练分类器，使用 {len(queries)} 个样本")
            train_model(temp_data_path, self.model_path)
            
            # 清理临时文件
            if os.path.exists(temp_data_path):
                try:
                    os.remove(temp_data_path)
                    logger.info(f"已清理临时训练数据文件: {temp_data_path}")
                except Exception as e:
                    logger.warning(f"清理临时文件时出错: {str(e)}")
            
            # 加载训练好的模型
            with open(self.model_path, "rb") as f:
                self.classifier = pickle.load(f)
            
            # 创建分类器集成
            self.classifier_integration = QueryClassifierIntegration(
                classifier=self.classifier,
                config=self.config
            )
            
            logger.info(f"分类器训练完成，模型已保存到: {self.model_path}")
            
            # 评估模型（可选）
            if self.classifier_config.get("evaluate_after_training", True):
                await self.evaluate(train_data_path)
            
            return True
        except Exception as e:
            logger.error(f"训练分类器时出错: {str(e)}")
            return False
    
    async def evaluate(self, eval_data_path: str) -> Dict[str, float]:
        """
        评估查询分类器性能
        
        Args:
            eval_data_path: 评估数据路径
            
        Returns:
            评估指标字典
        """
        if not os.path.exists(eval_data_path):
            logger.error(f"找不到评估数据: {eval_data_path}")
            return {}
        
        if self.classifier is None:
            logger.error("分类器尚未初始化或训练")
            return {}
        
        logger.info(f"开始评估查询分类器，使用数据: {eval_data_path}")
        
        try:
            # 加载评估数据
            with open(eval_data_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            
            # 提取查询和标签
            queries = [item.get("query", "") for item in eval_data if "query" in item]
            true_labels = [item.get("label", "") for item in eval_data if "label" in item]
            
            if len(queries) != len(true_labels) or len(queries) == 0:
                logger.error(f"评估数据不完整: {len(queries)} 个查询, {len(true_labels)} 个标签")
                return {}
            
            # 预测标签
            pred_labels = self.classifier.predict(queries)
            
            # 计算准确率
            correct = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
            accuracy = correct / len(true_labels) if true_labels else 0
            
            # 计算每个类别的精确率和召回率
            labels = set(true_labels)
            precision_recall = {}
            
            for label in labels:
                true_positives = sum(1 for true, pred in zip(true_labels, pred_labels) 
                                    if true == label and pred == label)
                false_positives = sum(1 for true, pred in zip(true_labels, pred_labels) 
                                     if true != label and pred == label)
                false_negatives = sum(1 for true, pred in zip(true_labels, pred_labels) 
                                     if true == label and pred != label)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precision_recall[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": sum(1 for true in true_labels if true == label)
                }
            
            # 汇总评估结果
            metrics = {
                "accuracy": accuracy,
                "class_metrics": precision_recall,
                "samples": len(queries)
            }
            
            # 保存评估结果
            eval_result_path = os.path.join(self.model_dir, "query_classifier_eval.json")
            with open(eval_result_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分类器评估完成，准确率: {accuracy:.4f}，结果已保存到: {eval_result_path}")
            
            return metrics
        except Exception as e:
            logger.error(f"评估分类器时出错: {str(e)}")
            return {}
    
    async def classify_query(self, query: str) -> Tuple[str, float]:
        """
        对查询进行分类
        
        Args:
            query: 查询文本
            
        Returns:
            分类标签和置信度
        """
        if self.classifier is None:
            if not self.load_model():
                logger.error("无法加载分类器模型")
                return "default", 0.0
        
        try:
            # 预测标签和置信度
            label, confidence = self.classifier.predict_with_confidence(query)
            logger.info(f"查询 '{query}' 被分类为 '{label}'，置信度: {confidence:.4f}")
            return label, confidence
        except Exception as e:
            logger.error(f"分类查询时出错: {str(e)}")
            return "default", 0.0
    
    async def route_query(self, query: str) -> Dict[str, Any]:
        """
        根据查询分类结果路由到合适的处理流程
        
        Args:
            query: 查询文本
            
        Returns:
            路由信息字典
        """
        if self.classifier_integration is None:
            if not self.load_model():
                logger.error("无法加载分类器集成")
                return {"route": "default", "params": {}}
        
        try:
            # 获取路由信息
            route_info = await self.classifier_integration.route_query(query)
            logger.info(f"查询 '{query}' 被路由到 '{route_info.get('route', 'default')}' 处理流程")
            return route_info
        except Exception as e:
            logger.error(f"路由查询时出错: {str(e)}")
            return {"route": "default", "params": {}}