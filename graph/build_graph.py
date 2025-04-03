# graph/build_graph.py

import json
import os
import pickle
from typing import Dict, List, Tuple, Set, Any, Optional

import networkx as nx
import numpy as np
import spacy
import yaml
from tqdm import tqdm

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.embedding import TextEmbedding


class DocumentGraphBuilder:
    """
    文档图构建器，负责从文档切分后的文本块创建节点和边
    """
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        初始化文档图构建器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 图构建参数
        self.similarity_threshold = self.config['graph']['similarity_threshold']
        self.entity_overlap_threshold = self.config['graph']['entity_overlap_threshold']
        self.max_neighbors = self.config['graph']['max_neighbors']
        self.connect_isolated = self.config['graph']['connect_isolated_nodes']
        
        # 初始化图
        self.graph = nx.Graph()
        
        # 初始化NLP模型用于实体识别
        self.nlp = spacy.load("en_core_web_sm")
        
        # 初始化嵌入模型
        self.embedder = TextEmbedding(config_path)
    
    def extract_entities(self, text: str) -> Set[str]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体集合
        """
        doc = self.nlp(text)
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                entities.add(ent.text.lower())
        return entities
    
    def calculate_entity_overlap(self, entities1: Set[str], entities2: Set[str]) -> float:
        """
        计算两个实体集合的重叠度
        
        Args:
            entities1: 第一个实体集合
            entities2: 第二个实体集合
            
        Returns:
            重叠度，范围[0, 1]
        """
        if not entities1 or not entities2:
            return 0.0
        
        intersection = entities1.intersection(entities2)
        union = entities1.union(entities2)
        
        return len(intersection) / len(union)
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            余弦相似度，范围[-1, 1]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def add_similarity_edges(self, documents: List[Dict[str, Any]], embeddings: Dict[str, np.ndarray]):
        """
        基于文本相似度添加边
        
        Args:
            documents: 文档列表
            embeddings: 文档ID到向量表示的映射
        """
        doc_ids = [doc["id"] for doc in documents]
        
        # 计算所有文档对之间的相似度
        print("Adding similarity edges...")
        for i, doc_id1 in enumerate(tqdm(doc_ids)):
            # 为每个节点找到最相似的max_neighbors个邻居
            similarities = []
            
            for j, doc_id2 in enumerate(doc_ids):
                if i == j:
                    continue
                
                # 计算余弦相似度
                similarity = self.calculate_cosine_similarity(
                    embeddings[doc_id1],
                    embeddings[doc_id2]
                )
                
                if similarity >= self.similarity_threshold:
                    similarities.append((doc_id2, similarity))
            
            # 按相似度排序并取前max_neighbors个
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = similarities[:self.max_neighbors]
            
            # 添加边
            for neighbor_id, similarity in top_neighbors:
                self.graph.add_edge(
                    doc_id1,
                    neighbor_id,
                    weight=similarity,
                    type="similarity"
                )
    
    def add_entity_edges(self, documents: List[Dict[str, Any]]):
        """
        基于实体共现添加边
        
        Args:
            documents: 文档列表
        """
        # 提取每个文档的实体
        doc_entities = {}
        print("Extracting entities...")
        for doc in tqdm(documents):
            doc_id = doc["id"]
            entities = self.extract_entities(doc["content"])
            doc_entities[doc_id] = entities
        
        # 计算实体重叠度并添加边
        print("Adding entity edges...")
        doc_ids = list(doc_entities.keys())
        for i, doc_id1 in enumerate(tqdm(doc_ids)):
            for j, doc_id2 in enumerate(doc_ids):
                if i >= j:  # 避免重复计算
                    continue
                
                # 计算实体重叠度
                overlap = self.calculate_entity_overlap(
                    doc_entities[doc_id1],
                    doc_entities[doc_id2]
                )
                
                if overlap >= self.entity_overlap_threshold:
                    # 检查是否已经存在相似度边
                    if self.graph.has_edge(doc_id1, doc_id2):
                        # 更新边属性
                        self.graph[doc_id1][doc_id2]["entity_overlap"] = overlap
                        self.graph[doc_id1][doc_id2]["type"] = "similarity+entity"
                    else:
                        # 添加新边
                        self.graph.add_edge(
                            doc_id1,
                            doc_id2,
                            weight=overlap,
                            type="entity",
                            entity_overlap=overlap
                        )
    
    def connect_isolated_nodes(self, documents: List[Dict[str, Any]], embeddings: Dict[str, np.ndarray]):
        """
        连接孤立节点
        
        Args:
            documents: 文档列表
            embeddings: 文档ID到向量表示的映射
        """
        if not self.connect_isolated:
            return
        
        # 获取孤立节点
        isolated_nodes = list(nx.isolates(self.graph))
        if not isolated_nodes:
            return
        
        print(f"Found {len(isolated_nodes)} isolated nodes, connecting...")
        
        # 构建文档ID到文档的映射
        id_to_doc = {doc["id"]: doc for doc in documents}
        
        # 对每个孤立节点，找到最相似的节点并连接
        for node_id in tqdm(isolated_nodes):
            # 计算与所有非孤立节点的相似度
            non_isolated_nodes = [n for n in self.graph.nodes() if n not in isolated_nodes]
            similarities = []
            
            for other_id in non_isolated_nodes:
                similarity = self.calculate_cosine_similarity(
                    embeddings[node_id],
                    embeddings[other_id]
                )
                similarities.append((other_id, similarity))
            
            # 按相似度排序并取前3个
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = similarities[:3]
            
            # 添加边
            for neighbor_id, similarity in top_neighbors:
                self.graph.add_edge(
                    node_id,
                    neighbor_id,
                    weight=similarity,
                    type="bridge"
                )
    
    def build_graph(self, documents_path: str, save_path: Optional[str] = None) -> nx.Graph:
        """
        构建文档图
        
        Args:
            documents_path: 文档路径，JSON格式
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            构建的图
        """
        # 加载文档
        with open(documents_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"Loaded {len(documents)} documents")
        
        # 为每个文档创建节点
        for doc in documents:
            self.graph.add_node(
                doc["id"],
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 计算文档嵌入
        embeddings = self.embedder.embed_documents(
            documents,
            cache_file=os.path.join(self.config['embedding']['cache_dir'], "base_embeddings.pkl")
        )
        
        # 添加相似度边
        self.add_similarity_edges(documents, embeddings)
        
        # 添加实体边
        self.add_entity_edges(documents)
        
        # 连接孤立节点
        self.connect_isolated_nodes(documents, embeddings)
        
        # 打印图统计信息
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # 保存图
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(self.graph, f)
            print(f"Graph saved to {save_path}")
        
        return self.graph


if __name__ == "__main__":
    # 示例用法
    builder = DocumentGraphBuilder("../config.yaml")
    graph = builder.build_graph(
        "../data/chunks.json",
        "../data/document_graph.pkl"
    )