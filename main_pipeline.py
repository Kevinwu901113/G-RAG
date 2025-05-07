#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRAG项目主流水线

该脚本实现了GRAG项目的主要流水线，包括四个独立步骤：
1. 文档预处理与图结构构建
2. 向量嵌入与索引构建
3. 查询分类器集成
4. 剪枝模块应用

每个步骤可以单独运行，并保存中间结果，方便后续只执行需要的部分以节省时间。
"""

import os
import sys
import yaml
import json
import asyncio
import argparse
from typing import Dict, List, Any, Optional, Tuple

# 导入项目模块
from grag.pipeline.document_processor import DocumentProcessor
from grag.pipeline.embedding_indexer import EmbeddingIndexer
from grag.pipeline.query_classifier_manager import QueryClassifierManager
from grag.pipeline.pruner_manager import PrunerManager
from grag.pipeline.query_processor import QueryProcessor
from grag.utils.logger_manager import configure_logging, get_logger

# 创建日志器
logger = get_logger("main_pipeline")

# 加载配置文件
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


async def run_document_processing(config: Dict[str, Any], data_dir: str) -> bool:
    """
    运行文档预处理与图结构构建步骤
    
    Args:
        config: 配置字典
        data_dir: 数据目录路径
        
    Returns:
        是否成功完成处理
    """
    logger.info("开始文档预处理与图结构构建步骤")
    
    try:
        # 创建文档处理器
        processor = DocumentProcessor(config)
        
        # 扫描并处理数据目录
        success = await processor.process_directory(data_dir)
        
        if success:
            logger.info("文档预处理与图结构构建步骤完成")
        else:
            logger.error("文档预处理与图结构构建步骤失败")
            
        return success
    except Exception as e:
        logger.error(f"文档预处理与图结构构建步骤出错: {str(e)}")
        return False


async def run_embedding_indexing(config: Dict[str, Any]) -> bool:
    """
    运行向量嵌入与索引构建步骤
    
    Args:
        config: 配置字典
        
    Returns:
        是否成功完成处理
    """
    logger.info("开始向量嵌入与索引构建步骤")
    
    try:
        # 创建嵌入索引器
        indexer = EmbeddingIndexer(config)
        
        # 构建索引
        success = await indexer.build_indices()
        
        if success:
            logger.info("向量嵌入与索引构建步骤完成")
        else:
            logger.error("向量嵌入与索引构建步骤失败")
            
        return success
    except Exception as e:
        logger.error(f"向量嵌入与索引构建步骤出错: {str(e)}")
        return False


async def run_query_classifier(config: Dict[str, Any], train_data_path: Optional[str] = None) -> bool:
    """
    运行查询分类器集成步骤
    
    Args:
        config: 配置字典
        train_data_path: 训练数据路径，如果为None则使用配置中的路径
        
    Returns:
        是否成功完成处理
    """
    logger.info("开始查询分类器集成步骤")
    
    try:
        # 创建查询分类器管理器
        classifier_manager = QueryClassifierManager(config)
        
        # 如果提供了训练数据，则训练分类器
        if train_data_path:
            success = await classifier_manager.train(train_data_path)
        else:
            # 否则只加载现有模型
            success = classifier_manager.load_model()
        
        if success:
            logger.info("查询分类器集成步骤完成")
        else:
            logger.error("查询分类器集成步骤失败")
            
        return success
    except Exception as e:
        logger.error(f"查询分类器集成步骤出错: {str(e)}")
        return False


async def run_pruner(config: Dict[str, Any], train_data_path: Optional[str] = None) -> bool:
    """
    运行剪枝模块应用步骤
    
    Args:
        config: 配置字典
        train_data_path: 训练数据路径，如果为None则使用配置中的路径
        
    Returns:
        是否成功完成处理
    """
    logger.info("开始剪枝模块应用步骤")
    
    try:
        # 创建剪枝管理器
        pruner_manager = PrunerManager(config)
        
        # 如果提供了训练数据，则训练剪枝模型
        if train_data_path:
            success = await pruner_manager.train(train_data_path)
        else:
            # 否则只应用现有模型进行剪枝
            success = await pruner_manager.apply_pruning()
        
        if success:
            logger.info("剪枝模块应用步骤完成")
        else:
            logger.error("剪枝模块应用步骤失败")
            
        return success
    except Exception as e:
        logger.error(f"剪枝模块应用步骤出错: {str(e)}")
        return False


async def run_query_processor(config: Dict[str, Any], query: str) -> Tuple[str, List[str]]:
    """
    运行查询处理
    
    Args:
        config: 配置字典
        query: 查询文本
        
    Returns:
        回答和来源列表
    """
    logger.info(f"处理查询: {query}")
    
    try:
        # 创建查询处理器
        processor = QueryProcessor(config)
        
        # 处理查询
        answer, sources = await processor.process_query(query)
        
        logger.info("查询处理完成")
        return answer, sources
    except Exception as e:
        logger.error(f"查询处理出错: {str(e)}")
        return f"处理查询时出错: {str(e)}", []


async def run_pipeline(args):
    """
    运行完整流水线或指定步骤
    
    Args:
        args: 命令行参数
    """
    # 加载配置
    config = load_config(args.config)
    
    # 配置日志
    log_dir = os.path.join(config["storage"]["working_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    configure_logging(log_dir=log_dir)
    
    # 根据参数决定运行哪些步骤
    if args.step == "all" or args.step == "document":
        success = await run_document_processing(config, args.data_dir)
        if not success and args.step == "document":
            return
    
    if args.step == "all" or args.step == "embedding":
        success = await run_embedding_indexing(config)
        if not success and args.step == "embedding":
            return
    
    if args.step == "all" or args.step == "classifier":
        success = await run_query_classifier(config, args.train_data)
        if not success and args.step == "classifier":
            return
    
    if args.step == "all" or args.step == "pruner":
        success = await run_pruner(config, args.train_data)
        if not success and args.step == "pruner":
            return
    
    # 如果提供了查询，则处理查询
    if args.query:
        answer, sources = await run_query_processor(config, args.query)
        print(f"\n问题: {args.query}")
        print(f"回答: {answer}")
        
        if sources:
            print("\n来源:")
            for source in sources:
                print(f"- {source}")


def main():
    """
    主函数，解析命令行参数并运行流水线
    """
    parser = argparse.ArgumentParser(description="GRAG项目主流水线")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径，默认为config.yaml")
    parser.add_argument("--step", type=str, default="all", 
                        choices=["all", "document", "embedding", "classifier", "pruner"], 
                        help="要运行的步骤，默认为all（全部步骤）")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录路径，默认为./data")
    parser.add_argument("--train_data", type=str, help="训练数据路径，用于查询分类器或剪枝模型训练")
    parser.add_argument("--query", type=str, help="要处理的查询文本")
    
    args = parser.parse_args()
    
    # 运行流水线
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()