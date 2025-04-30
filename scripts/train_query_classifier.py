#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查询分类器训练脚本

该脚本用于训练多标签文本分类器，用于判断查询策略和精度需求。
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grag.rag.query_classifier import train_model, load_data
from grag.utils.logger_manager import configure_logging, get_logger

# 配置日志
logger = get_logger("train_query_classifier")

def main():
    """
    主函数，用于解析参数并训练模型
    """
    parser = argparse.ArgumentParser(description="查询分类器训练脚本")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="训练数据路径，JSONL格式，每行包含query、query_strategy和precision_required字段")
    parser.add_argument("--output_dir", type=str, default="./query_classifier_model", 
                        help="模型输出目录，默认为 ./query_classifier_model")
    parser.add_argument("--batch_size", type=int, default=8, help="训练批次大小，默认为8")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数，默认为5")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--sample_data", action="store_true", help="显示数据样例并退出")
    
    args = parser.parse_args()
    
    # 配置日志
    configure_logging(log_dir=args.log_dir)
    
    # 记录训练开始时间
    start_time = datetime.now()
    logger.info(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 如果只是查看数据样例
        if args.sample_data:
            texts, labels = load_data(args.data_path)
            print(f"\n数据集包含 {len(texts)} 条记录")
            
            if len(texts) > 0:
                print("\n数据样例:")
                for i in range(min(5, len(texts))):
                    query_strategy = "hybrid" if labels[i][0] == 1 else "noRAG"
                    precision_required = "yes" if labels[i][1] == 1 else "no"
                    print(f"\n样例 {i+1}:")
                    print(f"查询: {texts[i]}")
                    print(f"查询策略: {query_strategy} (标签值: {labels[i][0]})")
                    print(f"精度要求: {precision_required} (标签值: {labels[i][1]})")
            
            return 0
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 保存训练配置
        config = {
            "data_path": args.data_path,
            "output_dir": args.output_dir,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "training_time": start_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # 训练模型
        logger.info(f"开始训练模型，数据路径: {args.data_path}, 输出目录: {args.output_dir}")
        logger.info(f"训练参数: batch_size={args.batch_size}, epochs={args.epochs}")
        
        model, tokenizer = train_model(
            data_path=args.data_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )
        
        # 记录训练结束时间
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总训练时间: {duration}")
        
        print(f"\n模型训练完成！")
        print(f"模型已保存到: {os.path.abspath(args.output_dir)}")
        print(f"总训练时间: {duration}")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"错误: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())