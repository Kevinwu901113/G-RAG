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
        
        # 3. Get entity descriptions
        entity_descriptions = []
        for entity_name in entity_names:
            entity_data = await self.graph_storage.get_node(entity_name)
            if entity_data:
                entity_descriptions.append(
                    [
                        entity_name,
                        entity_data.get("description", ""),
                    ]
                )

        # 4. Truncate descriptions to fit token limit
        truncated_descriptions = truncate_list_by_token_size(
            entity_descriptions,
            lambda x: x[0] + x[1],
            max_token,
        )

        # 5. Format as CSV
        header = ["ID", "Entity Name and Description"]
        csv_data = [header] + [
            [str(i + 1), f"{desc[0]}: {desc[1]}"]
            for i, desc in enumerate(truncated_descriptions)
        ]
        return list_of_list_to_csv(csv_data)

    async def _get_global_context(
        self, query: str, keywords: Dict[str, List[str]], top_k: int, max_token: int
    ) -> str:
        """
        Get global context based on relationship search.
        """
        # 1. Search for relationships using vector search
        relationship_results = await self.relationship_vector_storage.query(
            query, top_k=top_k
        )
        relationships = [
            (result["src_id"], result["tgt_id"]) for result in relationship_results
        ]
        
        # 2. Add relationships involving entities from keywords
        all_keywords = keywords.get("high_level_keywords", []) + keywords.get("low_level_keywords", [])
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
        
        # 3. Get all edges from the graph that mention any of the keywords in their description
        # This is a more comprehensive approach to find all relevant edges
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

        # 5. Get relationship descriptions
        relationship_descriptions = []
        for src_id, tgt_id in relationships:
            edge_data = await self.graph_storage.get_edge(src_id, tgt_id)
            if edge_data:
                relationship_descriptions.append(
                    [
                        f"{src_id} -> {tgt_id}",
                        edge_data.get("description", ""),
                    ]
                )

        # 6. Truncate descriptions to fit token limit
        truncated_descriptions = truncate_list_by_token_size(
            relationship_descriptions,
            lambda x: x[0] + x[1],
            max_token,
        )

        # 7. Format as CSV
        header = ["ID", "Relationship and Description"]
        csv_data = [header] + [
            [str(i + 1), f"{desc[0]}: {desc[1]}"]
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
        
        # 1. Prepare prompt
        entity_extraction_prompt = PROMPTS["entity_extraction"].format(
            entity_types=", ".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
            input_text=content,
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        )

        # 2. Extract entities and relationships
        logger.info("正在使用LLM提取实体和关系...")
        extraction_response = await self.llm_model_func(
            entity_extraction_prompt, **self.llm_model_kwargs
        )
        logger.info("LLM提取完成，开始解析结果")

        # 3. Parse response
        entities = []
        relationships = []
        content_keywords = []

        # Split by record delimiter
        records = extraction_response.split(PROMPTS["DEFAULT_RECORD_DELIMITER"])
        total_records = len(records)
        progress_bar = create_progress_bar(total_records, "解析实体和关系")
        
        for i, record in enumerate(records):
            record = record.strip()
            if not record:
                progress_bar.update(i+1, "跳过空记录")
                continue

            # Parse record
            parts = record.split(PROMPTS["DEFAULT_TUPLE_DELIMITER"])
            if len(parts) < 2:
                progress_bar.update(i+1, "跳过无效记录")
                continue

            record_type = parts[0].strip('(")')
            if record_type == "entity":
                if len(parts) >= 5:
                    entity_type = parts[1].strip()
                    entity_name = parts[2].strip()
                    name_type = parts[3].strip()
                    attributes = parts[4].strip()
                    entities.append(
                        {
                            "entity_type": name_type,
                            "name": entity_name,
                            "description": attributes,
                            "source_id": doc_id,
                        }
                    )
                    progress_bar.update(i+1, f"解析实体: {entity_name}")
            elif record_type == "relationship":
                if len(parts) >= 6:
                    source_entity = parts[1].strip()
                    target_entity = parts[2].strip()
                    relationship_description = parts[3].strip()
                    relationship_keywords = parts[4].strip()
                    relationship_strength = parts[5].strip().rstrip(")")
                    try:
                        strength = float(relationship_strength)
                    except ValueError:
                        strength = 5.0  # Default strength
                    relationships.append(
                        {
                            "source": source_entity,
                            "target": target_entity,
                            "description": relationship_description,
                            "keywords": relationship_keywords,
                            "weight": strength,
                            "source_id": doc_id,
                        }
                    )
                    progress_bar.update(i+1, f"解析关系: {source_entity} -> {target_entity}")
            elif record_type == "content_keywords":
                if len(parts) >= 2:
                    keywords = parts[1].strip().rstrip(")")
                    content_keywords.append(keywords)
                    progress_bar.update(i+1, "解析关键词")
            else:
                progress_bar.update(i+1, f"未知记录类型: {record_type}")
        
        progress_bar.finish()
        logger.info(f"解析完成，发现 {len(entities)} 个实体和 {len(relationships)} 个关系")

        # 4. Store entities and relationships
        if entities or relationships:
            storage_progress = create_progress_bar(
                len(entities) + len(relationships), 
                "存储实体和关系"
            )
            
            # 存储实体
            for i, entity in enumerate(entities):
                await self.graph_storage.upsert_node(entity["name"], entity)
                storage_progress.update(i+1, f"存储实体: {entity['name']}")
            
            # 存储关系
            offset = len(entities)
            for i, relationship in enumerate(relationships):
                await self.graph_storage.upsert_edge(
                    relationship["source"],
                    relationship["target"],
                    {
                        "description": relationship["description"],
                        "keywords": relationship["keywords"],
                        "weight": relationship["weight"],
                        "source_id": relationship["source_id"],
                    },
                )
                storage_progress.update(offset+i+1, f"存储关系: {relationship['source']} -> {relationship['target']}")
            
            storage_progress.finish()
            logger.info("实体和关系存储完成")

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
