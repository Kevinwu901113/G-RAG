import os
import json
from typing import List, Dict, Any, Tuple

from ..utils.common import logger, compute_mdhash_id
from ..utils.file_scanner import parse_hotpotqa_json
from .entity_extraction import process_and_store_entities_relationships


async def process_hotpotqa_dataset(rag, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    处理HotpotQA数据集并构建知识图谱
    
    Args:
        rag: LightRAG实例
        file_path: HotpotQA数据集文件路径
        
    Returns:
        (文档ID, 问答对列表)
    """
    logger.info(f"开始处理HotpotQA数据集: {file_path}")
    
    # 解析HotpotQA数据集
    kg_nodes, qa_pairs = parse_hotpotqa_json(file_path)
    
    if not kg_nodes:
        logger.warning(f"从 {file_path} 中未提取到任何知识图谱节点")
        return None, []
    
    # 将所有节点内容合并为一个文档
    combined_content = "\n\n".join([f"[{node['title']}] {node['content']}" for node in kg_nodes])
    
    # 生成文档ID
    doc_id = compute_mdhash_id(combined_content, prefix="hotpotqa-")
    
    # 索引文档
    logger.info(f"开始索引HotpotQA数据，共 {len(kg_nodes)} 个节点")
    await rag.index_document(
        content=combined_content,
        doc_id=doc_id,
        chunk_size=1000,  # 使用默认的分块大小
        chunk_overlap=200  # 使用默认的重叠大小
    )
    
    # 从kg_nodes构建实体和关系
    entities, relationships = convert_kg_nodes_to_entities_relationships(kg_nodes, doc_id, file_path)
    logger.info(f"从HotpotQA数据中提取了 {len(entities)} 个实体和 {len(relationships)} 个关系")
    
    # 处理并存储实体和关系
    if entities or relationships:
        await process_and_store_entities_relationships(
            entities=entities,
            relationships=relationships,
            knowledge_graph_inst=rag.graph_storage,
            entity_vdb=rag.entity_vector_storage,
            relationships_vdb=rag.relationship_vector_storage,
            force_llm_summary_on_merge=6,  # 默认合并阈值
            llm_model_func=rag.llm_model_func,
            llm_model_kwargs=rag.llm_model_kwargs,
        )
        logger.info(f"HotpotQA数据的实体和关系已存储到图谱中")
    
    logger.info(f"HotpotQA数据集 {file_path} 处理完成，文档ID: {doc_id}")
    return doc_id, qa_pairs


def convert_kg_nodes_to_entities_relationships(kg_nodes: List[Dict[str, Any]], doc_id: str, file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将HotpotQA知识图谱节点转换为实体和关系
    
    Args:
        kg_nodes: 知识图谱节点列表
        doc_id: 文档ID
        file_path: 文件路径
        
    Returns:
        (实体列表, 关系列表)
    """
    entities = []
    relationships = []
    
    # 标题实体映射，用于避免重复创建相同标题的实体
    title_entities = {}
    
    # 首先创建所有标题实体
    for node in kg_nodes:
        title = node.get('title', '')
        if title and title not in title_entities:
            # 为每个标题创建一个实体
            entity_id = compute_mdhash_id(title, prefix="ent-")
            title_entity = {
                "id": entity_id,
                "name": title,
                "type": "主题",  # 标题作为主题类型
                "entity_type": "主题",  # 兼容现有代码
                "description": f"HotpotQA数据集中的主题: {title}",
                "source_id": doc_id,
                "file_path": file_path
            }
            entities.append(title_entity)
            title_entities[title] = title_entity
    
    # 然后为每个句子创建实体，并与其标题建立关系
    for i, node in enumerate(kg_nodes):
        title = node.get('title', '')
        content = node.get('content', '')
        is_supporting = node.get('is_supporting', False)
        
        if not content or not title:
            continue
            
        # 为句子创建实体
        sentence_id = f"{title}-sentence-{i}"
        entity_id = compute_mdhash_id(sentence_id, prefix="ent-")
        
        # 确定实体类型
        entity_type = "支持事实" if is_supporting else "句子"
        
        sentence_entity = {
            "id": entity_id,
            "name": sentence_id,
            "type": entity_type,
            "entity_type": entity_type,  # 兼容现有代码
            "description": content,
            "source_id": doc_id,
            "file_path": file_path
        }
        entities.append(sentence_entity)
        
        # 创建句子与标题之间的关系
        if title in title_entities:
            relationship = {
                "source": sentence_id,
                "target": title,
                "description": f"句子属于主题 {title}",
                "keywords": "属于,包含",
                "weight": 5.0,
                "source_id": doc_id,
                "file_path": file_path
            }
            relationships.append(relationship)
            
            # 如果是支持事实，添加一个额外的关系
            if is_supporting:
                support_relationship = {
                    "source": sentence_id,
                    "target": title,
                    "description": f"这是支持主题 {title} 的关键事实",
                    "keywords": "支持,关键,事实",
                    "weight": 8.0,  # 更高的权重
                    "source_id": doc_id,
                    "file_path": file_path
                }
                relationships.append(support_relationship)
    
    return entities, relationships