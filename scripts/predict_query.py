#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查询分类器推理脚本

该脚本用于加载训练好的查询分类器模型，并对输入的查询进行分类预测。
"""

import os
import sys
import argparse

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grag.rag.query_classifier import QueryClassifier
from grag.utils.logger_manager import configure_logging, get_logger

# 配置日志
logger = get_logger("predict_query")

def main():
    """
    主函数，用于加载模型并进行推理
    """
    parser = argparse.ArgumentParser(description="查询分类器推理脚本")
    parser.add_argument("--query", type=str, required=True, help="需要分类的查询文本")
    parser.add_argument("--model_path", type=str, default="./query_classifier_model", 
                        help="模型加载路径，默认为 ./query_classifier_model")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    
    args = parser.parse_args()
    
    # 配置日志
    configure_logging(log_dir=args.log_dir)
    
    try:
        # 加载分类器
        logger.info(f"从 {args.model_path} 加载模型")
        classifier = QueryClassifier(args.model_path)
        
        # 进行预测
        logger.info(f"对查询进行预测: {args.query}")
        result = classifier.predict(args.query)
        
        # 输出结果
        print("\n预测结果:")
        print(f"查询策略 (query_strategy): {result['query_strategy']}")
        print(f"精度要求 (precision_required): {result['precision_required']}")
        print("\n详细解释:")
        
        # 提供结果解释
        if result['query_strategy'] == "noRAG":
            print("- 该查询不需要使用RAG检索增强生成技术处理")
        else:  # hybrid
            print("- 该查询需要使用混合检索策略处理")
            
        if result['precision_required'] == "yes":
            print("- 该查询需要高精度支持")
        else:  # no
            print("- 该查询不需要特别的精度支持")
            
        logger.info(f"预测完成: {result}")
        
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        print(f"错误: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())