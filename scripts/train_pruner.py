#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import argparse
import asyncio
from typing import Dict, List, Tuple, Any

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grag.pruner import GraphPruner
from grag.utils.common import get_logger
from grag.core.base import QueryParam
from grag.rag.lightrag import LightRAG
from grag.core.llm import ollama_embedding

# 获取日志器
logger = get_logger("train_pruner")

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
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

# 生成模拟数据
def generate_mock_data(num_nodes: int = 100, 
                     embedding_dim: int = 128, 
                     edge_density: float = 0.1) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, str]], List[int]]:
    """
    生成模拟数据用于训练
    
    Args:
        num_nodes: 节点数量
        embedding_dim: 嵌入维度
        edge_density: 边密度（0-1之间）
        
    Returns:
        节点嵌入字典、边列表和边标签
    """
    # 生成节点ID和嵌入
    node_embeddings = {}
    for i in range(num_nodes):
        node_id = f"node_{i}"
        # 生成随机嵌入并归一化
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        node_embeddings[node_id] = embedding
    
    # 生成边
    edges = []
    edge_labels = []
    
    # 计算可能的边数量
    max_edges = int(num_nodes * (num_nodes - 1) * edge_density)
    
    # 随机生成边
    for _ in range(max_edges):
        src_idx = np.random.randint(0, num_nodes)
        tgt_idx = np.random.randint(0, num_nodes)
        
        # 避免自环
        if src_idx == tgt_idx:
            continue
            
        src_id = f"node_{src_idx}"
        tgt_id = f"node_{tgt_idx}"
        
        # 计算余弦相似度作为边的重要性指标
        src_emb = node_embeddings[src_id]
        tgt_emb = node_embeddings[tgt_id]
        similarity = np.dot(src_emb, tgt_emb)
        
        # 根据相似度决定边的标签（保留或剪枝）
        # 相似度高的边更可能保留
        label = 1 if similarity > 0.2 or np.random.random() < 0.3 else 0
        
        edges.append((src_id, tgt_id))
        edge_labels.append(label)
    
    logger.info(f"生成了 {len(node_embeddings)} 个节点和 {len(edges)} 条边")
    logger.info(f"保留边的比例: {sum(edge_labels)/len(edge_labels):.2f}")
    
    return node_embeddings, edges, edge_labels

# 从实际图中提取数据
async def extract_data_from_graph(config_path: str) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, str]]]:
    """
    从已有的知识图谱中提取数据
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        节点嵌入字典和边列表
    """
    # 加载配置
    config = load_config(config_path)
    
    # 配置工作目录
    working_dir = config["storage"]["working_dir"]
    
    # 确保工作目录存在
    if not os.path.exists(working_dir):
        raise FileNotFoundError(f"工作目录 {working_dir} 不存在，请先运行main.py索引文档")
    
    logger.info(f"从 {working_dir} 加载图数据")
    
    # 导入必要的模块
    import networkx as nx
    from grag.core.storage import NetworkXStorage
    
    # 加载图
    graph_file = os.path.join(working_dir, "graph_graph.graphml")
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"图文件 {graph_file} 不存在")
    
    # 加载图
    G = nx.read_graphml(graph_file)
    
    # 获取节点和边
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    logger.info(f"从图中加载了 {len(nodes)} 个节点和 {len(edges)} 条边")
    
    # 获取嵌入模型配置
    embed_config = config["embedding"]
    
    # 获取或生成节点嵌入
    node_embeddings = {}
    
    # 如果有实体向量存储，尝试从中获取嵌入
    entity_vector_file = os.path.join(working_dir, "entities_vectors.npy")
    entity_id_file = os.path.join(working_dir, "entities_ids.json")
    
    if os.path.exists(entity_vector_file) and os.path.exists(entity_id_file):
        import json
        # 加载实体ID
        with open(entity_id_file, 'r') as f:
            entity_ids = json.load(f)
        
        # 加载实体向量
        entity_vectors = np.load(entity_vector_file)
        
        # 构建嵌入字典
        for i, entity_id in enumerate(entity_ids):
            if i < len(entity_vectors):
                node_embeddings[entity_id] = entity_vectors[i]
    else:
        # 如果没有现成的嵌入，为每个节点生成随机嵌入
        embedding_dim = embed_config["embedding_dim"]
        for node in nodes:
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            node_embeddings[node] = embedding
    
    # 转换边格式
    edge_list = [(u, v) for u, v in edges]
    
    return node_embeddings, edge_list

async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练图剪枝模型")
    parser.add_argument('--config', type=str, default="config.yaml", help='配置文件路径')
    parser.add_argument('--use_real_data', action='store_true', help='是否使用实际图数据')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数')
    parser.add_argument('--no_amp', action='store_true', help='禁用自动混合精度训练')
    parser.add_argument('--top_k_ratio', type=float, default=0.7, help='剪枝时保留的边比例')
    parser.add_argument('--improved_labels', action='store_true', help='使用改进的标签生成方法')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取工作目录
    working_dir = config["storage"]["working_dir"]
    
    # 确保模型目录存在
    models_dir = os.path.join(working_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 模型保存路径
    model_path = os.path.join(models_dir, "prune_model.pt")
    
    # 获取嵌入维度
    embedding_dim = config["embedding"]["embedding_dim"]
    
    # 获取数据
    if args.use_real_data:
        logger.info("使用实际图数据")
        node_embeddings, edges = await extract_data_from_graph(args.config)
        
        # 构建图结构
        graph_structure = {}
        for src_id, tgt_id in edges:
            if src_id not in graph_structure:
                graph_structure[src_id] = []
            if tgt_id not in graph_structure:
                graph_structure[tgt_id] = []
            graph_structure[src_id].append(tgt_id)
            graph_structure[tgt_id].append(src_id)  # 假设是无向图
        
        # 使用改进的标签生成方法或随机生成标签
        if args.improved_labels:
            logger.info("使用改进的标签生成方法")
            edge_labels = generate_improved_labels(edges, node_embeddings, graph_structure)
        else:
            logger.info("使用随机生成的标签")
            edge_labels = [1 if np.random.random() < 0.7 else 0 for _ in range(len(edges))]
    else:
        logger.info("使用模拟数据")
        # 生成模拟数据
        node_embeddings, edges, edge_labels = generate_mock_data(
            num_nodes=200, 
            embedding_dim=embedding_dim, 
            edge_density=0.05
        )
    
    # 初始化图剪枝器
    pruner = GraphPruner(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout
    )
    
    # 训练模型
    logger.info("开始训练模型...")
    history = pruner.train(
        node_embeddings=node_embeddings,
        edges=edges,
        edge_labels=edge_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp
    )
    
    # 保存模型
    pruner.save_model(model_path)
    
    # 测试剪枝
    logger.info("测试剪枝功能...")
    pruned_edges = pruner.prune(
        node_embeddings=node_embeddings,
        edges=edges,
        threshold=0.5,
        top_k_ratio=args.top_k_ratio,
        pruning_strategy='top_k'
    )
    
    # 输出剪枝结果统计
    logger.info(f"原始边数: {len(edges)}")
    logger.info(f"剪枝后边数: {len(pruned_edges)}")
    logger.info(f"剪枝率: {1 - len(pruned_edges)/len(edges):.2f}")
    
    # 输出一些保留的边示例
    if pruned_edges:
        logger.info("保留边示例:")
        for i, (src, tgt, prob) in enumerate(pruned_edges[:5]):
            logger.info(f"  边 {i+1}: {src} -> {tgt}, 保留概率: {prob:.4f}")

if __name__ == "__main__":
    asyncio.run(main())

# 为实际数据生成更好的标签
def generate_improved_labels(edges: List[Tuple[str, str]], 
                         node_embeddings: Dict[str, np.ndarray],
                         graph_structure: Dict[str, List[str]] = None) -> List[int]:
    """
    使用结构启发式指标生成边的标签
    
    Args:
        edges: 边列表，每条边为源节点ID和目标节点ID的元组
        node_embeddings: 节点ID到嵌入的映射
        graph_structure: 图结构，节点ID到其邻居节点ID列表的映射
        
    Returns:
        边标签列表，1表示保留，0表示剪枝
    """
    # 如果没有提供图结构，则创建一个
    if graph_structure is None:
        graph_structure = {}
        # 从边列表构建图结构
        for src_id, tgt_id in edges:
            if src_id not in graph_structure:
                graph_structure[src_id] = []
            if tgt_id not in graph_structure:
                graph_structure[tgt_id] = []
            graph_structure[src_id].append(tgt_id)
            graph_structure[tgt_id].append(src_id)  # 假设是无向图
    
    # 计算节点度数
    node_degrees = {node: len(neighbors) for node, neighbors in graph_structure.items()}
    
    # 计算全局统计信息
    avg_degree = np.mean([len(neighbors) for neighbors in graph_structure.values()]) if graph_structure else 0
    
    # 生成标签
    edge_labels = []
    for src_id, tgt_id in edges:
        # 获取节点的邻居集合
        src_neighbors = set(graph_structure.get(src_id, []))
        tgt_neighbors = set(graph_structure.get(tgt_id, []))
        
        # 计算结构特征
        # 1. 共同邻居数量
        common_neighbors = len(src_neighbors.intersection(tgt_neighbors))
        
        # 2. Jaccard相似度 = |A∩B| / |A∪B|
        union_size = len(src_neighbors.union(tgt_neighbors))
        jaccard_similarity = common_neighbors / max(union_size, 1)  # 避免除零错误
        
        # 3. 节点度数
        src_degree = node_degrees.get(src_id, 0)
        tgt_degree = node_degrees.get(tgt_id, 0)
        
        # 4. 嵌入相似度
        if src_id in node_embeddings and tgt_id in node_embeddings:
            src_emb = node_embeddings[src_id]
            tgt_emb = node_embeddings[tgt_id]
            embedding_similarity = np.dot(src_emb, tgt_emb)
        else:
            embedding_similarity = 0
        
        # 5. Adamic-Adar指数
        adamic_adar = 0
        for common_neighbor in src_neighbors.intersection(tgt_neighbors):
            neighbor_degree = node_degrees.get(common_neighbor, 0)
            if neighbor_degree > 1:  # 避免除零
                adamic_adar += 1.0 / np.log(neighbor_degree)
        
        # 综合多个指标确定边的标签
        # 设置阈值，满足任一条件即保留
        should_keep = False
        
        # 条件1: 共同邻居数量较多
        if common_neighbors >= 2:
            should_keep = True
        
        # 条件2: Jaccard相似度较高
        if jaccard_similarity >= 0.2:
            should_keep = True
        
        # 条件3: 嵌入相似度较高
        if embedding_similarity >= 0.5:
            should_keep = True
        
        # 条件4: Adamic-Adar指数较高
        if adamic_adar >= 1.0:
            should_keep = True
        
        # 条件5: 连接重要节点（高度数节点）
        if src_degree > avg_degree * 2 or tgt_degree > avg_degree * 2:
            # 高度数节点的连接更可能重要，但需要有一定的共同邻居或相似度
            if common_neighbors > 0 or jaccard_similarity > 0.1 or embedding_similarity > 0.3:
                should_keep = True
        
        edge_labels.append(1 if should_keep else 0)
    
    # 输出标签统计
    positive_ratio = sum(edge_labels) / len(edge_labels)
    logger.info(f"生成的标签中正样本比例: {positive_ratio:.2f}")
    
    return edge_labels