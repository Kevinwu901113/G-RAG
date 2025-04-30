#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查询分类器示例数据生成脚本

该脚本用于生成示例训练数据，方便测试查询分类器的功能。
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grag.utils.logger_manager import configure_logging, get_logger

# 配置日志
logger = get_logger("generate_sample_data")

# 示例查询模板
SAMPLE_QUERIES = {
    # noRAG查询示例
    "noRAG": [
        "你能告诉我今天的天气吗？",
        "写一首关于春天的诗",
        "如何做红烧肉？",
        "讲个笑话",
        "你是谁？",
        "解释一下量子力学",
        "翻译这句话成英文",
        "计算123乘以456",
        "给我讲个故事",
        "如何学习编程？"
    ],
    # hybrid查询示例
    "hybrid": [
        "总结一下这篇文章的主要观点",
        "分析这个数据集的趋势",
        "比较这两种算法的优缺点",
        "解释这段代码的功能",
        "查找关于气候变化的最新研究",
        "提取这篇论文中的关键信息",
        "分析这个公司的财务状况",
        "总结这个项目的主要成果",
        "查找与人工智能相关的最新进展",
        "分析这个实验的结果"
    ]
}

# 示例查询变体，用于增加数据多样性
QUERY_VARIANTS = {
    "prefix": [
        "请", "麻烦", "能否", "希望你能", "我想要", "我需要", "帮我", "我想请你", "能够", ""
    ],
    "suffix": [
        "谢谢", "非常感谢", "感谢你的帮助", "这对我很重要", "急需", "尽快", ""
    ]
}

def generate_sample_query(query_type, precision_required):
    """
    生成示例查询
    
    Args:
        query_type: 查询类型，"noRAG"或"hybrid"
        precision_required: 是否需要精度支持，True或False
        
    Returns:
        生成的查询文本
    """
    # 随机选择一个基础查询
    base_query = random.choice(SAMPLE_QUERIES[query_type])
    
    # 随机添加前缀和后缀
    prefix = random.choice(QUERY_VARIANTS["prefix"])
    suffix = random.choice(QUERY_VARIANTS["suffix"]) if random.random() < 0.3 else ""
    
    # 对于需要精度支持的查询，添加相关词语
    precision_phrases = [
        "请确保准确性", "需要精确的结果", "请提供精确信息", "需要高精度的答案",
        "请务必准确", "精确度很重要", "需要详细准确的信息"
    ]
    
    precision_phrase = random.choice(precision_phrases) if precision_required and random.random() < 0.7 else ""
    
    # 组合查询
    if precision_phrase:
        if random.random() < 0.5:
            query = f"{prefix}{base_query}，{precision_phrase}{suffix}".strip()
        else:
            query = f"{prefix}{precision_phrase}，{base_query}{suffix}".strip()
    else:
        query = f"{prefix}{base_query}{suffix}".strip()
    
    return query

def main():
    """
    主函数，用于生成示例数据
    """
    parser = argparse.ArgumentParser(description="查询分类器示例数据生成脚本")
    parser.add_argument("--output_path", type=str, default="./data/query_classifier_sample_data.jsonl", 
                        help="输出文件路径，默认为 ./data/query_classifier_sample_data.jsonl")
    parser.add_argument("--num_samples", type=int, default=100, 
                        help="生成的样本数量，默认为100")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    
    args = parser.parse_args()
    
    # 配置日志
    configure_logging(log_dir=args.log_dir)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        logger.info(f"开始生成 {args.num_samples} 条示例数据")
        
        # 生成样本数据
        samples = []
        for _ in range(args.num_samples):
            # 随机决定查询类型和精度需求
            query_type = "hybrid" if random.random() < 0.5 else "noRAG"
            precision_required = "yes" if random.random() < 0.5 else "no"
            
            # 生成查询文本
            query = generate_sample_query(query_type, precision_required == "yes")
            
            # 创建样本
            sample = {
                "query": query,
                "query_strategy": query_type,
                "precision_required": precision_required
            }
            
            samples.append(sample)
        
        # 写入文件
        with open(args.output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logger.info(f"成功生成 {len(samples)} 条示例数据，已保存到 {args.output_path}")
        
        # 打印示例
        print(f"\n成功生成 {len(samples)} 条示例数据，已保存到 {args.output_path}")
        print("\n数据样例:")
        for i in range(min(5, len(samples))):
            print(f"\n样例 {i+1}:")
            print(f"查询: {samples[i]['query']}")
            print(f"查询策略: {samples[i]['query_strategy']}")
            print(f"精度要求: {samples[i]['precision_required']}")
        
    except Exception as e:
        logger.error(f"生成数据过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"错误: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())