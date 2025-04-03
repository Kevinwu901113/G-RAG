# utils/embedding.py

import os
import pickle
from typing import List, Dict, Any, Union

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TextEmbedding:
    """
    文本嵌入类，负责将文本转换为向量表示
    """
    
    def __init__(self, config_path: str = "../config.yaml"):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 检查并下载模型
        model_name = self.config['embedding']['model_name']
        local_path = self.config['embedding']['local_model_path']
        
        if local_path and not os.path.exists(local_path):
            print(f"本地模型未找到，开始下载 {model_name}...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                model = SentenceTransformer(model_name)
                # 修正：使用 save 方法而不是 save_to_directory
                model.save(local_path)
                print(f"模型已下载并保存到 {local_path}")
            except Exception as e:
                raise RuntimeError(f"模型下载失败: {e}")
        
        # 初始化模型
        self.model = SentenceTransformer(local_path)
        
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        将单个文本转换为向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            文本的向量表示
        """
        # 确保文本是字符串类型
        if not isinstance(text, str):
            text = str(text)
            
        # 使用与 embed_texts 相同的参数，但不显示进度条
        return self.model.encode(text, show_progress_bar=True, convert_to_numpy=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        将多个文本转换为向量表示
        
        Args:
            texts: 输入文本列表
            
        Returns:
            文本的向量表示列表
        """
        return self.model.encode(texts, show_progress_bar=True)
    
    def embed_documents(self, documents: List[Dict[str, Any]], cache_file: Union[str, None] = None) -> Dict[str, np.ndarray]:
        """
        将文档列表转换为向量表示
        
        Args:
            documents: 文档列表，每个文档包含id和content字段
            cache_file: 缓存文件路径，如果为None则不缓存
            
        Returns:
            文档ID到向量表示的映射
        """
        # 检查缓存
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 提取文本
        doc_ids = [doc["id"] for doc in documents]
        texts = [doc["content"] for doc in documents]
        
        # 计算嵌入
        embeddings = self.embed_texts(texts)
        
        # 构建映射
        id_to_embedding = {doc_id: embedding for doc_id, embedding in zip(doc_ids, embeddings)}
        
        # 缓存结果
        if cache_file:
            with open(cache_file, 'wb') as f:
                pickle.dump(id_to_embedding, f)
        
        return id_to_embedding
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        将查询文本转换为向量表示
        
        Args:
            query: 查询文本
            
        Returns:
            查询的向量表示
        """
        return self.embed_text(query)


if __name__ == "__main__":
    # 示例用法
    import json
    
    # 加载文档
    with open("../data/chunks.json", 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # 初始化嵌入模型
    embedder = TextEmbedding("../config.yaml")
    
    # 计算嵌入
    embeddings = embedder.embed_documents(
        documents,
        cache_file="../data/embeddings/base_embeddings.pkl"
    )
    
    print(f"Embedded {len(embeddings)} documents")
    print(f"Embedding dimension: {next(iter(embeddings.values())).shape}")