import os
import re
import json
import asyncio
import argparse
from typing import List, Tuple, Dict, Any

from grag.core.base import QueryParam
from grag.rag.lightrag import LightRAG
from grag.core.llm import ollama_model_complete, ollama_embedding
from grag.utils.common import wrap_embedding_func_with_attrs, get_logger
from grag.core.storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage

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

# 获取日志器
logger = get_logger("grag")

async def setup_rag() -> LightRAG:
    """
    设置并初始化RAG系统，直接使用已有的存储数据
    
    Returns:
        初始化好的LightRAG实例
    """
    # 加载配置
    CONFIG = load_config()
    
    # 配置工作目录
    WORKING_DIR = CONFIG["storage"]["working_dir"]
    
    # 确保工作目录存在
    if not os.path.exists(WORKING_DIR):
        raise FileNotFoundError(f"工作目录 {WORKING_DIR} 不存在，请先运行main.py索引文档")
    
    # 获取嵌入模型配置
    embed_config = CONFIG["embedding"]
    
    # 创建嵌入函数
    embedding_func = wrap_embedding_func_with_attrs(
        embedding_dim=embed_config["embedding_dim"],
        max_token_size=embed_config["max_token_size"]
    )(
        lambda texts: ollama_embedding(
            texts, embed_model=embed_config["model_name"], host=embed_config["host"]
        )
    )
    
    # 创建全局配置
    global_config = {
        "working_dir": WORKING_DIR,
        "embedding_batch_num": CONFIG["storage"]["embedding_batch_num"],
        "cosine_better_than_threshold": CONFIG["query"]["cosine_better_than_threshold"],
        "llm_model_name": CONFIG["llm"]["model_name"],
        "graph_file": os.path.join(WORKING_DIR, "graph_graph.graphml"),
    }
    
    # 创建存储实例
    doc_full_storage = JsonKVStorage(
        namespace="full_docs",
        global_config=global_config,
        embedding_func=embedding_func
    )
    
    doc_chunks_storage = JsonKVStorage(
        namespace="text_chunks",
        global_config=global_config,
        embedding_func=embedding_func
    )
    
    chunks_vector_storage = NanoVectorDBStorage(
        namespace="chunks",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"tokens", "chunk_order_index", "full_doc_id"}
    )
    
    entity_vector_storage = NanoVectorDBStorage(
        namespace="entities",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"entity_name"}
    )
    
    relationship_vector_storage = NanoVectorDBStorage(
        namespace="relationships",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"src_id", "tgt_id"}
    )
    
    graph_storage = NetworkXStorage(
        namespace="graph",
        global_config=global_config,
        embedding_func=embedding_func
    )
    
    # 创建LightRAG实例
    rag = LightRAG(
        doc_full_storage=doc_full_storage,
        doc_chunks_storage=doc_chunks_storage,
        chunks_vector_storage=chunks_vector_storage,
        entity_vector_storage=entity_vector_storage,
        relationship_vector_storage=relationship_vector_storage,
        graph_storage=graph_storage,
        llm_model_func=ollama_model_complete,
        llm_model_kwargs={
            "host": CONFIG["llm"]["host"], 
            "options": CONFIG["llm"]["options"],
            "hashing_kv": doc_full_storage
        },
        embedding_func=embedding_func,
    )
    
    return rag

async def query_and_process(
    rag: LightRAG, 
    question: str, 
    mode: str = None,
    config: Dict[str, Any] = None
) -> Tuple[str, List[str]]:
    """
    查询RAG系统并处理结果
    
    Args:
        rag: LightRAG实例
        question: 查询问题
        mode: 查询模式 (local, global, hybrid, naive)
        config: 配置字典
        
    Returns:
        回答和来源列表
    """
    # 如果未提供配置，加载默认配置
    if config is None:
        config = load_config()
    
    # 如果未指定模式，使用配置中的默认模式
    if mode is None:
        mode = config["query"]["default_mode"]
    
    # 设置查询参数
    param = QueryParam(
        mode=mode,
        top_k=config["query"]["top_k"],
        max_token_for_text_unit=config["query"]["max_token_for_text_unit"],
        max_token_for_global_context=config["query"]["max_token_for_global_context"],
        max_token_for_local_context=config["query"]["max_token_for_local_context"]
    )
    
    # 获取上下文
    context = await rag.aquery(question, param=QueryParam(mode=mode, only_need_context=True))
    
    # 获取回答
    answer = await rag.aquery(question, param=param)
    
    # 处理来源
    sources = []
    if mode != "naive":
        lines = context.split('\n')
        pattern = re.compile(r'^\d+,')
        current_source = ""
        for line in lines:
            if pattern.match(line):
                if current_source:
                    sources.append(current_source)
                current_source = line
            elif current_source:
                current_source += " " + line
        
        if current_source:
            sources.append(current_source)
    
    return answer, sources

async def process_json_questions(json_file_path: str, mode: str = None):
    """
    处理JSON文件中的问题列表
    
    Args:
        json_file_path: JSON文件路径
        mode: 查询模式
    """
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误: JSON文件 {json_file_path} 不存在")
        return
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件 {json_file_path}")
        return
    
    # 验证JSON格式
    if not isinstance(questions, list):
        print(f"错误: JSON文件应包含问题列表")
        return
    
    # 设置RAG系统
    rag = await setup_rag()
    config = load_config()
    
    # 处理每个问题
    for i, question in enumerate(questions):
        if isinstance(question, str):
            query_text = question
        elif isinstance(question, dict) and 'question' in question:
            query_text = question['question']
        else:
            print(f"警告: 跳过无效问题格式: {question}")
            continue
        
        print(f"\n问题 {i+1}/{len(questions)}: {query_text}")
        print(f"模式: {mode if mode else config['query']['default_mode']}")
        
        # 查询并处理结果
        answer, sources = await query_and_process(rag, query_text, mode, config)
        
        # 输出结果
        print(f"回答: {answer}")
        
        if sources:
            print("\n来源:")
            for source in sources:
                print(f"- {source}")
        
        print("\n" + "-"*50)

async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GRAG查询工具")
    parser.add_argument('--mode', type=str, 
                        choices=['local', 'global', 'hybrid', 'naive'], 
                        help='选择检索模式')
    parser.add_argument('--json', type=str, help='JSON问题文件路径')
    parser.add_argument('--question', type=str, help='直接指定查询问题')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    
    # 如果未指定模式，使用配置中的默认模式
    mode = args.mode if args.mode else config["query"]["default_mode"]
    
    # 如果指定了JSON文件，处理文件中的问题
    if args.json:
        await process_json_questions(args.json, mode)
        return
    
    # 如果指定了问题，直接查询
    if args.question:
        question = args.question
    else:
        # 默认问题
        question = "公明镇镇长有谁担任过?"
    
    # 设置RAG系统
    rag = await setup_rag()
    
    # 查询并处理结果
    print(f"\n问题: {question}")
    print(f"模式: {mode}")
    
    answer, sources = await query_and_process(rag, question, mode, config)
    
    # 输出结果
    print(f"回答: {answer}")
    
    if sources:
        print("\n来源:")
        for source in sources:
            print(f"- {source}")

if __name__ == "__main__":
    asyncio.run(main())