# generator/generate_answer.py

import os
from typing import Dict, List, Any, Optional

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AnswerGenerator:
    """
    答案生成器，负责使用检索到的文档和用户查询生成最终答案
    """
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        初始化答案生成器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 生成参数
        self.model_name = self.config['generation']['model_name']
        self.max_length = self.config['generation']['max_length']
        self.num_beams = self.config['generation']['num_beams']
        self.temperature = self.config['generation']['temperature']
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
    
    def format_input(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        格式化输入文本，将查询和检索到的文档组合成模型输入
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            
        Returns:
            格式化的输入文本
        """
        # 基本格式："question: {query} context: {doc1} {doc2} ..."
        context = " ".join([doc["content"] for doc in documents])
        return f"question: {query} context: {context}"
    
    def generate(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        生成答案
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            
        Returns:
            生成的答案
        """
        # 格式化输入
        input_text = self.format_input(query, documents)
        
        # 编码输入
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        # 生成答案
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.max_length,
            num_beams=self.num_beams,
            temperature=self.temperature,
            early_stopping=True
        )
        
        # 解码输出
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer


class FiDGenerator(AnswerGenerator):
    """
    基于Fusion-in-Decoder (FiD) 的答案生成器
    FiD将每个检索到的文档单独编码，然后在解码器中融合
    """
    
    def format_input(self, query: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        格式化输入文本，为每个文档创建单独的输入
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            
        Returns:
            格式化的输入文本列表
        """
        # 为每个文档创建单独的输入
        inputs = []
        for doc in documents:
            input_text = f"question: {query} context: {doc['content']}"
            inputs.append(input_text)
        
        return inputs
    
    def generate(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        生成答案
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            
        Returns:
            生成的答案
        """
        # 格式化输入
        input_texts = self.format_input(query, documents)
        
        # 编码每个输入
        all_inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 生成答案
        # 注意：标准的Transformers库不直接支持FiD，这里是简化实现
        # 实际FiD需要修改模型架构，这里我们使用标准T5模型的简化版本
        outputs = self.model.generate(
            all_inputs.input_ids,
            attention_mask=all_inputs.attention_mask,
            max_length=self.max_length,
            num_beams=self.num_beams,
            temperature=self.temperature,
            early_stopping=True
        )
        
        # 解码输出
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer


def get_generator(config_path: str = "../config.yaml") -> AnswerGenerator:
    """
    根据配置获取适当的生成器
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        答案生成器实例
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 根据模型类型选择生成器
    model_name = config['generation']['model_name']
    
    # 如果是T5系列模型，使用标准生成器
    if "t5" in model_name.lower():
        return AnswerGenerator(config_path)
    # 如果是FiD模型，使用FiD生成器
    elif "fid" in model_name.lower():
        return FiDGenerator(config_path)
    # 默认使用标准生成器
    else:
        return AnswerGenerator(config_path)


if __name__ == "__main__":
    # 示例用法
    import json
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from retrieval.faiss_index import FaissRetriever
    
    # 初始化检索器
    retriever = FaissRetriever("../config.yaml")
    retriever.load_index()
    
    # 初始化生成器
    generator = get_generator("../config.yaml")
    
    # 测试查询
    query = "示例查询"
    
    # 检索文档
    documents = retriever.retrieve(query)
    
    # 生成答案
    answer = generator.generate(query, documents)
    
    print(f"Query: {query}")
    print(f"Answer: {answer}")