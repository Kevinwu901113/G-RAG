#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查询分类器训练测试脚本

该脚本用于测试查询分类器的训练功能，不添加新功能，仅调用项目已有的功能。
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from grag.rag.query_classifier import train_model, load_data
import json
from grag.utils.logger_manager import configure_logging, get_logger

# 配置日志
logger = get_logger("test_query_classifier")

def main():
    """
    主函数，用于解析参数并测试训练模型功能
    """
    parser = argparse.ArgumentParser(description="查询分类器训练测试脚本")
    parser.add_argument("--data_path", type=str, default="./train_data/default_query_samples.json", 
                        help="训练数据路径，JSONL格式，每行包含query、query_strategy和precision_required字段")
    parser.add_argument("--output_dir", type=str, default="./test_model_output", 
                        help="模型输出目录，默认为 ./test_model_output")
    parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小，默认为4")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数，默认为2")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--show_data", action="store_true", help="显示数据样例")
    
    args = parser.parse_args()
    
    # 配置日志
    configure_logging(log_dir=args.log_dir)
    
    logger.info("开始测试查询分类器训练功能")
    
    try:
        # 确保数据路径存在
        if not os.path.exists(args.data_path):
            logger.error(f"数据文件不存在: {args.data_path}")
            print(f"错误: 数据文件不存在: {args.data_path}")
            return 1
        
        # 加载数据
        # 尝试先以JSON数组格式加载数据
        try:
            with open(args.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                # 处理JSON数组格式
                texts = []
                labels = []
                
                for item in data:
                    query = item.get("query", "")
                    label = item.get("label", "norag")
                    
                    # 将标签转换为数字标签
                    query_strategy_label = 1 if label.lower() == "hybrid" else 0
                    precision_required_label = 0  # 默认为no
                    
                    texts.append(query)
                    labels.append([query_strategy_label, precision_required_label])
                
                logger.info(f"成功以JSON数组格式加载数据: {len(texts)} 条记录")
            else:
                # 如果不是数组格式，则尝试使用原始的load_data函数
                texts, labels = load_data(args.data_path)
                logger.info(f"成功加载数据: {len(texts)} 条记录")
        except json.JSONDecodeError:
            # 如果JSON解析失败，则尝试使用原始的load_data函数
            texts, labels = load_data(args.data_path)
            logger.info(f"成功加载数据: {len(texts)} 条记录")
        
        # 如果需要显示数据样例
        if args.show_data and len(texts) > 0:
            print(f"\n数据集包含 {len(texts)} 条记录")
            print("\n数据样例:")
            for i in range(min(3, len(texts))):
                query_strategy = "hybrid" if labels[i][0] == 1 else "noRAG"
                precision_required = "yes" if labels[i][1] == 1 else "no"
                print(f"\n样例 {i+1}:")
                print(f"查询: {texts[i]}")
                print(f"查询策略: {query_strategy} (标签值: {labels[i][0]})")
                print(f"精度要求: {precision_required} (标签值: {labels[i][1]})")
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 记录训练开始时间
        start_time = datetime.now()
        logger.info(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n开始训练模型...")
        print(f"数据路径: {args.data_path}")
        print(f"输出目录: {args.output_dir}")
        print(f"批次大小: {args.batch_size}")
        print(f"训练轮数: {args.epochs}")
        
        # 如果成功加载了数据，则创建临时JSONL文件
        if len(texts) > 0:
            temp_data_path = os.path.join(args.output_dir, "temp_train_data.jsonl")
            os.makedirs(os.path.dirname(temp_data_path), exist_ok=True)
            
            with open(temp_data_path, "w", encoding="utf-8") as f:
                for i, (query, label) in enumerate(zip(texts, labels)):
                    query_strategy = "hybrid" if label[0] == 1 else "noRAG"
                    precision_required = "yes" if label[1] == 1 else "no"
                    f.write(json.dumps({"query": query, "query_strategy": query_strategy, "precision_required": precision_required}) + "\n")
            
            logger.info(f"已创建临时JSONL格式训练数据: {temp_data_path}")
            
            # 调用训练函数，使用临时数据文件
            model, tokenizer = train_model(
                data_path=temp_data_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_epochs=args.epochs
            )
            
            # 清理临时文件
            try:
                os.remove(temp_data_path)
                logger.info(f"已清理临时训练数据文件")
            except Exception as e:
                logger.warning(f"清理临时文件时出错: {str(e)}")
        else:
            logger.warning("没有加载到有效数据，跳过训练")
            model, tokenizer = None, None
        
        # 记录训练结束时间
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总训练时间: {duration}")
        
        print(f"\n模型训练测试完成！")
        print(f"模型已保存到: {os.path.abspath(args.output_dir)}")
        print(f"总训练时间: {duration}")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"错误: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())