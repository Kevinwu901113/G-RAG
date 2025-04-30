import os
import json
from typing import List, Dict, Any, Tuple
from ..utils.common import logger


def scan_data_directory(data_dir: str = "data") -> Dict[str, List[str]]:
    """
    扫描数据目录，根据文件类型分类文件
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        按文件类型分类的文件路径字典
    """
    # 确保数据目录存在
    if not os.path.exists(data_dir):
        logger.warning(f"数据目录 {data_dir} 不存在，正在创建...")
        os.makedirs(data_dir, exist_ok=True)
        return {}
    
    # 分类文件
    file_types = {
        "docx": [],
        "json": [],
        "txt": [],
        "other": []
    }
    
    # 遍历数据目录中的所有文件
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower().lstrip('.')
            
            # 根据文件扩展名分类
            if file_ext in file_types:
                file_types[file_ext].append(file_path)
            else:
                file_types["other"].append(file_path)
    
    # 记录扫描结果
    for file_type, files in file_types.items():
        if files:
            logger.info(f"找到 {len(files)} 个 {file_type} 文件")
    
    return file_types


def parse_hotpotqa_json(file_path: str, max_samples: int = 100) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    解析HotpotQA数据集JSON文件
    
    Args:
        file_path: JSON文件路径
        max_samples: 最大处理样本数，默认为100，防止内存溢出
        
    Returns:
        (知识图谱节点列表, 问答对列表)
    """
    # 读取JSON文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 限制处理的样本数量，防止内存溢出
        if len(data) > max_samples:
            logger.warning(f"HotpotQA数据集包含 {len(data)} 个样本，限制处理前 {max_samples} 个样本以防内存溢出")
            data = data[:max_samples]
    except Exception as e:
        logger.error(f"解析JSON文件 {file_path} 失败: {e}")
        return [], []
    
    # 存储知识图谱节点和问答对
    kg_nodes = []
    qa_pairs = []
    
    # 遍历数据集中的每个样本
    for item in data:
        # 提取问题和答案
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        # 如果没有问题或答案，跳过
        if not question or not answer:
            continue
        
        # 保存问答对
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
        
        # 提取上下文
        supporting_facts = item.get("supporting_facts", [])
        context = item.get("context", [])
        
        # 处理上下文，将每个句子作为知识图谱节点
        # 限制每个样本的上下文数量，防止内存溢出
        max_context_items = 10
        if len(context) > max_context_items:
            context = context[:max_context_items]
            
        for title_sentences in context:
            if len(title_sentences) != 2:
                continue
                
            title = title_sentences[0]
            sentences = title_sentences[1]
            
            # 限制每个标题下的句子数量
            max_sentences = 20
            if len(sentences) > max_sentences:
                sentences = sentences[:max_sentences]
            
            # 将每个句子作为一个节点
            for i, sentence in enumerate(sentences):
                # 创建节点
                node = {
                    "content": sentence,
                    "title": title,
                    "sentence_id": i,
                    "source": os.path.basename(file_path),
                    "is_supporting": False
                }
                
                # 检查是否为支持事实
                for fact in supporting_facts:
                    if fact[0] == title and fact[1] == i:
                        node["is_supporting"] = True
                        break
                
                kg_nodes.append(node)
    
    logger.info(f"从 {file_path} 中提取了 {len(kg_nodes)} 个知识图谱节点和 {len(qa_pairs)} 个问答对")
    return kg_nodes, qa_pairs