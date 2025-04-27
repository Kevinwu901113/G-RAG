import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np

from ..core.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    QueryParam,
    TextChunkSchema,
)
from ..core.prompt import PROMPTS
from ..utils.common import (
    compute_mdhash_id,
    convert_response_to_json,
    encode_string_by_tiktoken,
    list_of_list_to_csv,
    logger,
    process_combine_contexts,
    truncate_list_by_token_size,
    create_progress_bar,
    set_logger,
)
from .entity_extraction import (
    extract_entities_and_relationships,
    process_and_store_entities_relationships,
)


@dataclass
class LightRAG:
    """
    LightRAG is a lightweight RAG implementation that supports both local and global context.
    """

    # Storage
    doc_full_storage: BaseKVStorage
    doc_chunks_storage: BaseKVStorage
    chunks_vector_storage: BaseVectorStorage
    entity_vector_storage: BaseVectorStorage
    relationship_vector_storage: BaseVectorStorage
    graph_storage: BaseGraphStorage

    # LLM
    llm_model_func: Callable
    llm_model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Prompt template
    prompt_template: Optional[str] = None

    # Cache
    cache_seed: int = 42
    cache_capacity: int = 1000

    # Embedding
    embedding_func: Optional[Callable] = None
    
    def __post_init__(self):
        # Get global_config from one of the storage objects
        self.global_config = self.doc_full_storage.global_config

    # Async
    async def aquery(
        self, query: str, param: QueryParam = None, **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Query the RAG system.
        """
        if param is None:
            param = QueryParam()

        # 1. Extract keywords from query
        keywords_extraction_prompt = PROMPTS["keywords_extraction"].format(
            query=query,
        )
        keywords_extraction_response = await self.llm_model_func(
            keywords_extraction_prompt, **self.llm_model_kwargs
        )
        try:
            # 处理LLM返回内容可能被markdown代码块包裹的情况
            cleaned_response = keywords_extraction_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # 提取JSON部分，忽略注释
            json_match = re.search(r'({[\s\S]*?})', cleaned_response)
            if json_match:
                json_str = json_match.group(1)
                keywords = json.loads(json_str)
            else:
                # 如果没有找到JSON格式，尝试直接解析
                keywords = json.loads(cleaned_response)
                
            # 检查返回内容是否包含所需字段，否则补全
            if not isinstance(keywords, dict):
                raise ValueError("LLM返回内容不是字典类型")
            if "high_level_keywords" not in keywords:
                keywords["high_level_keywords"] = []
            if "low_level_keywords" not in keywords:
                keywords["low_level_keywords"] = []
        except Exception as e:
            logger.error(
                f"Failed to parse keywords extraction response: {keywords_extraction_response}\nError: {e}"
            )
            keywords = {
                "high_level_keywords": [],
                "low_level_keywords": [],
            }

        # 2. Get context based on mode
        if param.mode == "local":
            context = await self._get_local_context(
                query, keywords, param.top_k, param.max_token_for_local_context
            )
        elif param.mode == "global":
            context = await self._get_global_context(
                query, keywords, param.top_k, param.max_token_for_global_context
            )
        elif param.mode == "hybrid":
            local_context = await self._get_local_context(
                query, keywords, param.top_k, param.max_token_for_local_context
            )
            global_context = await self._get_global_context(
                query, keywords, param.top_k, param.max_token_for_global_context
            )
            context = process_combine_contexts(global_context, local_context)
        elif param.mode == "naive":
            context = await self._get_naive_context(
                query, keywords, param.top_k, param.max_token_for_text_unit
            )
        else:
            raise ValueError(f"Invalid mode: {param.mode}")

        if param.only_need_context:
            return context

        # 3. Generate response
        if param.mode == "naive":
            prompt = PROMPTS["naive_rag_response"].format(
                query=query,
                response_type=param.response_type, 
                content_data=context
            )
        else:
            prompt = PROMPTS["rag_response"].format(
                query=query,
                response_type=param.response_type, 
                context_data=context
            )

        response = await self.llm_model_func(prompt, **self.llm_model_kwargs)
        return response

    async def _get_local_context(
        self, query: str, keywords: Dict[str, List[str]], top_k: int, max_token: int
    ) -> str:
        """
        Get local context based on entity search.
        """
        # 1. Search for entities using both query and keywords
        entity_results = await self.entity_vector_storage.query(query, top_k=top_k)
        entity_names = [result["entity_name"] for result in entity_results]
        
        # 2. Add entities from keywords if they exist in high_level_keywords or low_level_keywords
        all_keywords = keywords.get("high_level_keywords", []) + keywords.get("low_level_keywords", [])
        for keyword in all_keywords:
            # Check if the keyword is an entity name in the graph
            if await self.graph_storage.has_node(f'"{keyword}"'):
                if f'"{keyword}"' not in entity_names:
                    entity_names.append(f'"{keyword}"')
        
        # 3. 尝试从查询中识别实体类型
        query_entity_types = self._extract_entity_types_from_query(query, all_keywords)
        
        # 4. Get entity descriptions with additional information
        entity_details = []
        for entity_name in entity_names:
            entity_data = await self.graph_storage.get_node(entity_name)
            if entity_data:
                # 获取实体的id、name、type和description
                entity_id = entity_data.get("id", "")
                entity_type = entity_data.get("type", entity_data.get("entity_type", "UNKNOWN"))
                entity_description = entity_data.get("description", "")
                
                # 计算实体类型与查询类型的匹配度
                type_relevance = 1.0
                if query_entity_types and entity_type in query_entity_types:
                    type_relevance = 2.0  # 提高匹配类型的权重
                
                entity_details.append({
                    "name": entity_name,
                    "id": entity_id,
                    "type": entity_type,
                    "description": entity_description,
                    "relevance": type_relevance  # 用于排序
                })
        
        # 5. 根据类型相关性排序实体
        entity_details.sort(key=lambda x: x["relevance"], reverse=True)
# 6. 构建丰富的实体描述
        entity_descriptions = []
        for entity in entity_details:
            entity_name = entity['name'].replace('"', '')
            formatted_description = (
                f"[{entity['type']}] {entity_name}: {entity['description']}"
            )
            entity_descriptions.append([entity["id"], formatted_description])

        # 7. Truncate descriptions to fit token limit
        truncated_descriptions = truncate_list_by_token_size(
            entity_descriptions, lambda x: x[1], max_token
        )

        # 8. Format as CSV
        header = ["ID", "Entity Information"]
        csv_data = [header] + [
            [str(i + 1), desc[1]]
            for i, desc in enumerate(truncated_descriptions)
        ]
        return list_of_list_to_csv(csv_data)
    
    def _extract_entity_types_from_query(self, query: str, keywords: List[str]) -> List[str]:
        """
        从查询和关键词中提取可能的实体类型
        
        Args:
            query: 查询字符串
            keywords: 关键词列表
            
        Returns:
            可能的实体类型列表
        """
        entity_types = []
        
        # 检查查询中是否包含实体类型相关的词
        type_keywords = {
            "组织名称": ["组织", "公司", "机构", "团体", "委员会", "部门"],
            "个人姓名": ["人", "谁", "名字", "姓名"],
            "地理位置": ["哪里", "地方", "位置", "地点", "在哪", "哪儿"],
            "事件": ["事件", "发生", "什么事", "活动"],
            "时间": ["时间", "何时", "什么时候", "年", "月", "日", "几点"],
            "职位": ["职位", "职务", "担任", "工作", "岗位"],
            "金额": ["多少钱", "价格", "费用", "金额", "成本"],
            "面积": ["多大", "面积", "平方", "亩"],
            "人数": ["多少人", "人数", "数量", "几个"]
        }
        
        # 检查查询中是否包含类型关键词
        for entity_type, type_words in type_keywords.items():
            for word in type_words:
                if word in query:
                    entity_types.append(entity_type)
                    break
        
        # 检查关键词中是否包含类型信息
        for keyword in keywords:
            for entity_type, type_words in type_keywords.items():
                for word in type_words:
                    if word in keyword:
                        entity_types.append(entity_type)
                        break
        
        return list(set(entity_types))  # 去重

    async def _get_global_context(
        self, query: str, keywords: Dict[str, List[str]], top_k: int, max_token: int
    ) -> str:
        """
        Get global context based on relationship search.
        """
        # 1. 尝试从查询中识别实体类型
        all_keywords = keywords.get("high_level_keywords", []) + keywords.get("low_level_keywords", [])
        query_entity_types = self._extract_entity_types_from_query(query, all_keywords)
        
        # 2. Search for relationships using vector search
        relationship_results = await self.relationship_vector_storage.query(
            query, top_k=top_k
        )
        relationships = [
            (result["src_id"], result["tgt_id"]) for result in relationship_results
        ]
        
        # 3. Add relationships involving entities from keywords
        for keyword in all_keywords:
            keyword_with_quotes = f'"{keyword}"'
            
            # Check if the keyword is an entity name in the graph
            if await self.graph_storage.has_node(keyword_with_quotes):
                # Get all edges connected to this entity
                node_edges = await self.graph_storage.get_node_edges(keyword_with_quotes)
                if node_edges:
                    for edge in node_edges:
                        if edge not in relationships:
                            relationships.append(edge)
        
        # 4. Get all edges from the graph that mention any of the keywords in their description
        try:
            # Get all edges from the graph
            import networkx as nx
            import os
            
            # Load the graph from the graphml file
            graph_file = os.path.join(self.global_config["working_dir"], f"graph_{self.graph_storage.namespace}.graphml")
            if os.path.exists(graph_file):
                graph = nx.read_graphml(graph_file)
                
                # Find edges that mention any of the keywords in their description
                for keyword in all_keywords:
                    for src_id, tgt_id, edge_data in graph.edges(data=True):
                        description = edge_data.get("description", "")
                        if keyword.lower() in description.lower() and (src_id, tgt_id) not in relationships:
                            relationships.append((src_id, tgt_id))
        except Exception as e:
            logger.error(f"Error while searching for edges with keyword in description: {e}")

        # 5. 获取关系详细信息，包括实体类型和描述
        relationship_details = []
        for src_id, tgt_id in relationships:
            edge_data = await self.graph_storage.get_edge(src_id, tgt_id)
            if edge_data:
                # 获取源实体和目标实体的信息
                src_entity = await self.graph_storage.get_node(src_id)
                tgt_entity = await self.graph_storage.get_node(tgt_id)
                
                src_type = src_entity.get("type", src_entity.get("entity_type", "UNKNOWN")) if src_entity else "UNKNOWN"
                tgt_type = tgt_entity.get("type", tgt_entity.get("entity_type", "UNKNOWN")) if tgt_entity else "UNKNOWN"
                
                # 计算关系与查询类型的相关性
                type_relevance = 1.0
                if query_entity_types:
                    if src_type in query_entity_types or tgt_type in query_entity_types:
                        type_relevance = 2.0  # 提高匹配类型的权重
                
                # 获取关系的关键词和描述
                keywords = edge_data.get("keywords", "")
                description = edge_data.get("description", "")
                
                relationship_details.append({
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "src_type": src_type,
                    "tgt_type": tgt_type,
                    "keywords": keywords,
                    "description": description,
                    "relevance": type_relevance  # 用于排序
                })
        
        # 6. 根据类型相关性排序关系
        relationship_details.sort(key=lambda x: x["relevance"], reverse=True)
        
        # 7. 构建丰富的关系描述
        relationship_descriptions = []
        for rel in relationship_details:
            src_name = rel["src_id"].replace('"', '')
            tgt_name = rel["tgt_id"].replace('"', '')
            formatted_description = (
                f"[{rel['src_type']}] {src_name} -> [{rel['tgt_type']}] {tgt_name}: "
                f"{rel['description']} (关键词: {rel['keywords']})"
            )
            relationship_descriptions.append([
                f"{rel['src_id']} -> {rel['tgt_id']}",
                formatted_description
            ])

        # 8. Truncate descriptions to fit token limit
        truncated_descriptions = truncate_list_by_token_size(
            relationship_descriptions,
            lambda x: x[0] + x[1],
            max_token,
        )

        # 9. Format as CSV
        header = ["ID", "Relationship Information"]
        csv_data = [header] + [
            [str(i + 1), desc[1]]
            for i, desc in enumerate(truncated_descriptions)
        ]
        return list_of_list_to_csv(csv_data)

    async def _get_naive_context(
        self, query: str, keywords: Dict[str, List[str]], top_k: int, max_token: int
    ) -> str:
        """
        Get context based on direct chunk search.
        """
        # 1. Search for chunks
        chunk_results = await self.chunks_vector_storage.query(query, top_k=top_k)
        chunk_ids = [result["id"] for result in chunk_results]

        # 2. Get chunk contents
        chunks = await self.doc_chunks_storage.get_by_ids(chunk_ids)
        if not chunks:
            return ""

        # 3. Truncate chunks to fit token limit
        truncated_chunks = truncate_list_by_token_size(
            chunks, lambda x: x["content"], max_token
        )

        # 4. Format as text
        return "\n\n".join([chunk["content"] for chunk in truncated_chunks])

    async def index_document(
        self,
        content: str,
        doc_id: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> str:
        """
        Index a document into the RAG system.
        """
        # 创建总体进度条
        total_steps = 5  # 总共5个主要步骤
        main_progress = create_progress_bar(total_steps, "文档索引总进度")
        
        # 1. Generate document ID if not provided
        if doc_id is None:
            doc_id = compute_mdhash_id(content, prefix="doc-")
        logger.info(f"开始索引文档，ID: {doc_id}")
        main_progress.update(1, "生成文档ID")

        # 2. Store full document
        logger.info("存储完整文档")
        await self.doc_full_storage.upsert({doc_id: {"content": content}})
        main_progress.update(2, "存储完整文档")

        # 3. Split document into chunks
        logger.info("分割文档为文本块")
        chunks = self._split_text(content, chunk_size, chunk_overlap)
        logger.info(f"文档分割完成，共 {len(chunks)} 个文本块")
        
        # 4. Store chunks
        chunk_progress = create_progress_bar(len(chunks), "存储文本块")
        chunk_data = {}
        for i, chunk in enumerate(chunks):
            chunk_id = compute_mdhash_id(chunk, prefix=f"{doc_id}-chunk-")
            chunk_tokens = len(encode_string_by_tiktoken(chunk))
            chunk_data[chunk_id] = {
                "content": chunk,
                "tokens": chunk_tokens,
                "chunk_order_index": i,
                "full_doc_id": doc_id,
            }
            chunk_progress.update(i+1, f"处理文本块 {i+1}/{len(chunks)}")

        await self.doc_chunks_storage.upsert(chunk_data)
        chunk_progress.finish()
        main_progress.update(3, "存储文本块")

        # 5. Extract entities and relationships
        logger.info("开始提取实体和关系")
        await self._extract_entities_and_relationships(content, doc_id)
        main_progress.update(4, "提取实体和关系")

        # 6. Commit changes
        logger.info("提交所有更改")
        await self.doc_full_storage.index_done_callback()
        await self.doc_chunks_storage.index_done_callback()
        await self.chunks_vector_storage.index_done_callback()
        await self.entity_vector_storage.index_done_callback()
        await self.relationship_vector_storage.index_done_callback()
        await self.graph_storage.index_done_callback()
        main_progress.update(5, "提交更改")
        
        # 完成进度条
        main_progress.finish()
        logger.info(f"文档索引完成，ID: {doc_id}")

        return doc_id

    async def _extract_entities_and_relationships(
        self, content: str, doc_id: str
    ) -> None:
        """
        Extract entities and relationships from document content.
        """
        logger.info(f"开始从文档 {doc_id} 提取实体和关系")
        
        # 1. 提取实体和关系
        entities, relationships = await extract_entities_and_relationships(
            content=content,
            chunk_key=doc_id,
            llm_model_func=self.llm_model_func,
            llm_model_kwargs=self.llm_model_kwargs,
            knowledge_graph_inst=self.graph_storage,
            entity_vdb=self.entity_vector_storage,
            relationships_vdb=self.relationship_vector_storage,
            file_path=doc_id,  # 使用文档ID作为文件路径
            max_gleaning=1,    # 默认只进行一次额外提取
        )
        
        logger.info(f"提取完成，共 {len(entities)} 个实体和 {len(relationships)} 个关系")
        
        # 2. 处理并存储实体和关系
        if entities or relationships:
            await process_and_store_entities_relationships(
                entities=entities,
                relationships=relationships,
                knowledge_graph_inst=self.graph_storage,
                entity_vdb=self.entity_vector_storage,
                relationships_vdb=self.relationship_vector_storage,
                force_llm_summary_on_merge=6,  # 默认合并阈值
                llm_model_func=self.llm_model_func,
                llm_model_kwargs=self.llm_model_kwargs,
            )
            
            logger.info("实体和关系处理与存储完成")

    def _split_text(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """
        Split text into chunks.
        """
        if not text:
            return []

        # Split by paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If paragraph is too large, split it further
            if paragraph_size > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    sentence_size = len(sentence)
                    if current_size + sentence_size <= chunk_size:
                        current_chunk.append(sentence)
                        current_size += sentence_size
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_size = sentence_size
            else:
                # If adding paragraph exceeds chunk size, start a new chunk
                if current_size + paragraph_size > chunk_size:
                    chunks.append(" ".join(current_chunk))
                    
                    # For overlap, keep some of the previous content
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_size = sum(len(p) for p in current_chunk)
                
                current_chunk.append(paragraph)
                current_size += paragraph_size

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
