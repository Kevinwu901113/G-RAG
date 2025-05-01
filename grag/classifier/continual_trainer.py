#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
持续学习模块

该模块用于定期更新已实现的多标签查询分类器。
通过读取新的用户交互数据或人工标注数据，对现有模型进行增量训练，
实现无中断地更新分类模型。
"""

import os
import sys
import json
import yaml
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from shutil import copytree

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入查询分类器相关模块
from grag.rag.query_classifier import QueryClassifierModel, QueryDataset, LABEL_MAPS
from grag.utils.logger_manager import get_logger
from transformers import BertTokenizer, Trainer, TrainingArguments

# 创建日志器
logger = get_logger("continual_trainer")

class ContinualTrainer:
    """
    持续学习训练器，用于定期更新查询分类器模型
    """
    def __init__(self, 
                 config_path: str = None,
                 buffer_path: str = "logs/continual_buffer.jsonl",
                 model_path: str = "./query_classifier_model",
                 backup_model_dir: str = "./model_backups",
                 batch_size: int = 8,
                 epochs: int = 3,
                 save_backup: bool = True):
        """
        初始化持续学习训练器
        
        Args:
            config_path: 配置文件路径，如果提供，将从中加载配置
            buffer_path: 新数据缓冲区路径，默认为logs/continual_buffer.jsonl
            model_path: 模型路径，默认为./query_classifier_model
            backup_model_dir: 模型备份目录，默认为./model_backups
            batch_size: 训练批次大小，默认为8
            epochs: 训练轮数，默认为3
            save_backup: 是否保存旧模型备份，默认为True
        """
        # 初始化默认配置
        self.buffer_path = buffer_path
        self.model_path = model_path
        self.backup_model_dir = backup_model_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_backup = save_backup
        
        # 如果提供了配置文件路径，从中加载配置
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # 确保模型备份目录存在
        if self.save_backup and not os.path.exists(self.backup_model_dir):
            os.makedirs(self.backup_model_dir, exist_ok=True)
    
    def _load_config(self, config_path: str):
        """
        从YAML配置文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 获取持续学习相关配置
            continual_config = config.get('continual_learning', {})
            
            # 更新配置
            self.buffer_path = continual_config.get('buffer_path', self.buffer_path)
            self.model_path = continual_config.get('model_path', self.model_path)
            self.backup_model_dir = continual_config.get('backup_model_dir', self.backup_model_dir)
            self.batch_size = continual_config.get('batch_size', self.batch_size)
            self.epochs = continual_config.get('epochs', self.epochs)
            self.save_backup = continual_config.get('save_backup', self.save_backup)
            
            logger.info(f"从配置文件 {config_path} 加载持续学习配置")
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            raise
    
    def _validate_data_format(self, data: Dict) -> bool:
        """
        验证数据格式是否符合要求
        
        Args:
            data: 数据字典
            
        Returns:
            是否符合要求
        """
        # 检查必要字段是否存在
        required_fields = ["query", "query_strategy", "precision_required"]
        for field in required_fields:
            if field not in data:
                logger.warning(f"数据缺少必要字段: {field}")
                return False
        
        # 检查query_strategy字段值是否有效
        if data["query_strategy"] not in LABEL_MAPS["query_strategy"]:
            logger.warning(f"无效的query_strategy值: {data['query_strategy']}")
            return False
        
        # 检查precision_required字段值是否有效
        if data["precision_required"] not in LABEL_MAPS["precision_required"]:
            logger.warning(f"无效的precision_required值: {data['precision_required']}")
            return False
        
        return True
    
    def _load_new_data(self) -> Tuple[List[str], List[List[int]]]:
        """
        从缓冲区加载新数据
        
        Returns:
            texts: 文本列表
            labels: 标签列表
        """
        texts = []
        labels = []
        
        if not os.path.exists(self.buffer_path):
            logger.warning(f"数据缓冲区文件不存在: {self.buffer_path}")
            return texts, labels
        
        logger.info(f"从 {self.buffer_path} 加载新数据")
        
        try:
            with open(self.buffer_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # 验证数据格式
                        if not self._validate_data_format(data):
                            logger.warning(f"第 {line_num} 行数据格式无效，已跳过")
                            continue
                        
                        query = data["query"]
                        query_strategy = data["query_strategy"]
                        precision_required = data["precision_required"]
                        
                        # 将文本标签转换为数字标签
                        query_strategy_label = LABEL_MAPS["query_strategy"][query_strategy]
                        precision_required_label = LABEL_MAPS["precision_required"][precision_required]
                        
                        texts.append(query)
                        labels.append([query_strategy_label, precision_required_label])
                    except json.JSONDecodeError:
                        logger.warning(f"第 {line_num} 行无法解析为JSON，已跳过: {line}")
                        continue
                    except Exception as e:
                        logger.warning(f"处理第 {line_num} 行时出错: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"加载新数据时出错: {str(e)}")
            raise
        
        logger.info(f"成功加载 {len(texts)} 条新数据")
        return texts, labels
    
    def _backup_model(self):
        """
        备份当前模型
        """
        if not self.save_backup:
            logger.info("已禁用模型备份，跳过备份步骤")
            return
        
        if not os.path.exists(self.model_path):
            logger.warning(f"模型路径不存在，无法备份: {self.model_path}")
            return
        
        # 创建备份目录（如果不存在）
        os.makedirs(self.backup_model_dir, exist_ok=True)
        
        # 生成备份文件夹名称，包含时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_model_dir, f"query_classifier_{timestamp}")
        
        try:
            # 复制整个模型目录
            copytree(self.model_path, backup_path)
            logger.info(f"已将模型备份到: {backup_path}")
        except Exception as e:
            logger.error(f"备份模型时出错: {str(e)}")
            raise
    
    def run_continual_learning(self):
        """
        运行持续学习过程
        
        Returns:
            是否成功更新模型
        """
        # 加载新数据
        texts, labels = self._load_new_data()
        
        # 如果没有新数据，则跳过训练
        if len(texts) == 0:
            logger.info("没有新数据，跳过训练")
            return False
        
        try:
            # 检查模型路径是否存在
            if not os.path.exists(self.model_path):
                logger.error(f"模型路径不存在: {self.model_path}")
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
            
            # 备份当前模型
            self._backup_model()
            
            # 加载现有模型和分词器
            logger.info(f"加载现有模型: {self.model_path}")
            model = QueryClassifierModel.from_pretrained(self.model_path)
            tokenizer = BertTokenizer.from_pretrained(self.model_path)
            
            # 创建数据集
            dataset = QueryDataset(texts, labels, tokenizer)
            
            # 设置训练参数
            training_args = TrainingArguments(
                output_dir=self.model_path,
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                logging_dir=f"{self.model_path}/logs",
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="no",
                load_best_model_at_end=False,
                save_total_limit=2,
                overwrite_output_dir=True,  # 覆盖输出目录
            )
            
            # 定义数据整理函数
            def data_collator(features):
                batch = {}
                batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
                batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
                if "token_type_ids" in features[0]:
                    batch["token_type_ids"] = torch.stack([f["token_type_ids"] for f in features])
                if "labels" in features[0]:
                    batch["labels"] = torch.stack([f["labels"] for f in features])
                return batch
            
            # 初始化训练器
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            
            # 开始训练
            logger.info("开始持续学习训练...")
            trainer.train()
            
            # 保存模型和分词器
            logger.info(f"保存更新后的模型到 {self.model_path}")
            model.save_pretrained(self.model_path)
            tokenizer.save_pretrained(self.model_path)
            
            logger.info("持续学习训练完成")
            return True
        except Exception as e:
            logger.error(f"持续学习训练过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def run_continual_learning(config_path: str = None, **kwargs):
    """
    运行持续学习过程的便捷函数
    
    Args:
        config_path: 配置文件路径
        **kwargs: 其他参数，将覆盖配置文件中的设置
        
    Returns:
        是否成功更新模型
    """
    try:
        # 创建持续学习训练器
        trainer = ContinualTrainer(config_path=config_path, **kwargs)
        
        # 运行持续学习过程
        return trainer.run_continual_learning()
    except Exception as e:
        logger.error(f"运行持续学习过程时出错: {str(e)}")
        return False


if __name__ == "__main__":
    # 如果直接运行此脚本，则执行持续学习过程
    import argparse
    
    parser = argparse.ArgumentParser(description="查询分类器持续学习脚本")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--buffer_path", type=str, help="新数据缓冲区路径")
    parser.add_argument("--model_path", type=str, help="模型路径")
    parser.add_argument("--backup_model_dir", type=str, help="模型备份目录")
    parser.add_argument("--batch_size", type=int, help="训练批次大小")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--no_backup", action="store_true", help="禁用模型备份")
    
    args = parser.parse_args()
    
    # 构建参数字典
    kwargs = {}
    if args.buffer_path:
        kwargs["buffer_path"] = args.buffer_path
    if args.model_path:
        kwargs["model_path"] = args.model_path
    if args.backup_model_dir:
        kwargs["backup_model_dir"] = args.backup_model_dir
    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    if args.epochs:
        kwargs["epochs"] = args.epochs
    if args.no_backup:
        kwargs["save_backup"] = False
    
    # 运行持续学习过程
    success = run_continual_learning(config_path=args.config, **kwargs)
    
    if success:
        print("持续学习过程成功完成，模型已更新")
    else:
        print("持续学习过程未能完成，请查看日志了解详情")
        sys.exit(1)