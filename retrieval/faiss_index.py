# retrieval/faiss_index.py

import os
import pickle
from typing import Dict, List, Tuple, Any, Optional

import faiss
import numpy as np
import yaml
from tqdm import tqdm

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.embedding import TextEmbedding


class FaissRetriever:
    """
    基于FAISS的检索器，支持原始嵌入和图增强嵌入的混合检索
    """
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        初始化检索器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 检索参数
        self.index_path = self.config['retrieval']['index_path']
        self.top_k = self.config['retrieval']['top_k']
        self.use_graph_enhanced = self.config['retrieval']['use_graph_enhanced']
        self.hybrid_weight = self.config['retrieval']['hybrid_weight']
        
        # 确保索引目录存在
        os.makedirs(self.index_path, exist_ok=True)
        
        # 初始化嵌入模型
        self.embedder = TextEmbedding(config_path)
        
        # FAISS索引
        self.base_index = None
        self.graph_index = None
        
        # ID映射
        self.id_to_index = {}
        self.index_to_id = {}
        
        # 文档内容
        self.documents = []
    
    def build_index(self, documents: List[Dict[str, Any]], 
                    base_embeddings: Dict[str, np.ndarray],
                    graph_embeddings: Optional[Dict[str, np.ndarray]] = None):
        """
        构建FAISS索引
        
        Args:
            documents: 文档列表
            base_embeddings: 基础嵌入，节点ID到向量的映射
            graph_embeddings: 图增强嵌入，节点ID到向量的映射，可选
        """
        # 保存文档
        self.documents = documents
        
        # 构建ID映射
        doc_ids = [doc["id"] for doc in documents]
        self.id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        self.index_to_id = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        
        # 提取嵌入向量
        base_vectors = np.stack([base_embeddings[doc_id] for doc_id in doc_ids])
        
        # 构建基础索引
        dimension = base_vectors.shape[1]
        self.base_index = faiss.IndexFlatIP(dimension)  # 内积相似度（余弦相似度需要归一化向量）
        # 将向量添加到索引中,base_vectors作为参数x传入
        self.base_index.add(x=base_vectors, n=base_vectors.shape[0])
        
        # 构建图增强索引（如果提供）
        if graph_embeddings is not None and self.use_graph_enhanced:
            graph_vectors = np.stack([graph_embeddings[doc_id] for doc_id in doc_ids])
            self.graph_index = faiss.IndexFlatIP(dimension)
            self.graph_index.add(x=graph_vectors, n=graph_vectors.shape[0])
        
        # 保存索引和映射
        self._save_index()
    
    def _save_index(self):
        """
        保存索引和映射
        """
        # 保存基础索引
        faiss.write_index(self.base_index, os.path.join(self.index_path, "base_index.faiss"))
        
        # 保存图增强索引（如果存在）
        if self.graph_index is not None:
            faiss.write_index(self.graph_index, os.path.join(self.index_path, "graph_index.faiss"))
        
        # 保存ID映射
        with open(os.path.join(self.index_path, "id_mapping.pkl"), 'wb') as f:
            pickle.dump({
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id
            }, f)
        
        # 保存文档
        with open(os.path.join(self.index_path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
    
    def load_index(self):
        """
        加载索引和映射
        """
        # 加载基础索引
        base_index_path = os.path.join(self.index_path, "base_index.faiss")
        if os.path.exists(base_index_path):
            self.base_index = faiss.read_index(base_index_path)
        else:
            raise FileNotFoundError(f"Base index not found at {base_index_path}")
        
        # 加载图增强索引（如果存在）
        graph_index_path = os.path.join(self.index_path, "graph_index.faiss")
        if os.path.exists(graph_index_path) and self.use_graph_enhanced:
            self.graph_index = faiss.read_index(graph_index_path)
        
        # 加载ID映射
        mapping_path = os.path.join(self.index_path, "id_mapping.pkl")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                mapping = pickle.load(f)
                self.id_to_index = mapping["id_to_index"]
                self.index_to_id = mapping["index_to_id"]
        else:
            raise FileNotFoundError(f"ID mapping not found at {mapping_path}")
        
        # 加载文档
        documents_path = os.path.join(self.index_path, "documents.pkl")
        if os.path.exists(documents_path):
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            raise FileNotFoundError(f"Documents not found at {documents_path}")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        检索与查询最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量，如果为None则使用配置中的值
            
        Returns:
            检索到的文档列表，按相关性排序
        """
        if top_k is None:
            top_k = self.top_k
        
        # 确保索引已加载
        if self.base_index is None:
            try:
                self.load_index()
            except Exception as e:
                raise RuntimeError("Failed to load FAISS index") from e
        
        # 确保索引有效
        if self.base_index is None:
            raise RuntimeError("FAISS base index is not initialized")
            
        # 编码查询
        query_vector = self.embedder.embed_query(query)
        query_vector = query_vector.reshape(1, -1)  # 调整为二维数组
        
        # 使用基础索引检索
        distances, labels = self.base_index.search(query_vector, top_k)
        distances, labels = distances[0], labels[0]  # 取第一行（只有一个查询）
        
        # 如果不使用图增强索引或图增强索引不存在，直接返回基础检索结果
        if not self.use_graph_enhanced or self.graph_index is None:
            return self._format_results(distances, labels)
        
        # 使用图增强索引检索
        graph_distances, graph_labels = self.graph_index.search(query_vector, top_k)
        graph_distances, graph_labels = graph_distances[0], graph_labels[0]  # 取第一行（只有一个查询）
        
        # 混合检索结果
        return self._hybrid_results(distances, labels, graph_distances, graph_labels)
    
    def _format_results(self, distances: np.ndarray, labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        格式化检索结果
        
        Args:
            distances: 相似度分数
            labels: 索引标签
            
        Returns:
            格式化的检索结果
        """
        results = []
        for score, idx in zip(distances, labels):
            if idx == -1:  # FAISS返回-1表示没有足够的匹配项
                continue
            
            doc_id = self.index_to_id[idx]
            doc = next((d for d in self.documents if d["id"] == doc_id), None)
            
            if doc:
                results.append({
                    "id": doc_id,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(score)
                })
        
        return results
    
    def _hybrid_results(self, base_distances: np.ndarray, base_labels: np.ndarray,
                        graph_distances: np.ndarray, graph_labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        混合基础检索和图增强检索的结果
        
        Args:
            base_distances: 基础检索的相似度分数
            base_labels: 基础检索的索引标签
            graph_distances: 图增强检索的相似度分数
            graph_labels: 图增强检索的索引标签
            
        Returns:
            混合后的检索结果
        """
        # 构建得分映射
        id_to_score = {}
        
        # 添加基础检索得分
        for score, idx in zip(base_distances, base_labels):
            if idx == -1:  # FAISS返回-1表示没有足够的匹配项
                continue
            
            doc_id = self.index_to_id[idx]
            id_to_score[doc_id] = (1 - self.hybrid_weight) * score
        
        # 添加图增强检索得分
        for score, idx in zip(graph_distances, graph_labels):
            if idx == -1:  # FAISS返回-1表示没有足够的匹配项
                continue
            
            doc_id = self.index_to_id[idx]
            if doc_id in id_to_score:
                id_to_score[doc_id] += self.hybrid_weight * score
            else:
                id_to_score[doc_id] = self.hybrid_weight * score
        
        # 按得分排序
        sorted_ids = sorted(id_to_score.keys(), key=lambda x: id_to_score[x], reverse=True)
        sorted_ids = sorted_ids[:self.top_k]  # 限制结果数量
        
        # 格式化结果
        results = []
        for doc_id in sorted_ids:
            doc = next((d for d in self.documents if d["id"] == doc_id), None)
            
            if doc:
                results.append({
                    "id": doc_id,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(id_to_score[doc_id])
                })
        
        return results


if __name__ == "__main__":
    # 示例用法
    import json
    
    # 加载文档
    with open("../data/chunks.json", 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # 加载基础嵌入
    with open("../data/embeddings/base_embeddings.pkl", 'rb') as f:
        base_embeddings = pickle.load(f)
    
    # 加载图增强嵌入（如果存在）
    graph_embeddings = None
    graph_embeddings_path = "../data/embeddings/graph_embeddings.pkl"
    if os.path.exists(graph_embeddings_path):
        with open(graph_embeddings_path, 'rb') as f:
            graph_embeddings = pickle.load(f)
    
    # 初始化检索器
    retriever = FaissRetriever("../config.yaml")
    
    # 构建索引
    retriever.build_index(documents, base_embeddings, graph_embeddings)
    
    # 测试检索
    query = "示例查询"
    results = retriever.retrieve(query)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} documents")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['id']} (Score: {result['score']:.4f})")
        print(f"Content: {result['content'][:100]}...")
        print()