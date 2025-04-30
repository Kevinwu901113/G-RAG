import os
import re
import json
import asyncio
import datetime
import yaml
from typing import List, Optional, Tuple, Dict, Any, Callable
import argparse

from grag.core.base import QueryParam
from grag.rag.lightrag import LightRAG
from grag.core.llm import ollama_model_complete, ollama_embedding
from grag.utils.common import wrap_embedding_func_with_attrs, set_logger, create_progress_bar
from grag.utils.logger_manager import get_logger, configure_logging
from grag.core.storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
from grag.utils.file_scanner import scan_data_directory
from grag.rag.hotpotqa_processor import process_hotpotqa_dataset

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
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

# 加载配置
CONFIG = load_config()

# 配置工作目录和日志
WORKING_DIR = CONFIG["storage"]["working_dir"]
LOG_DIR = os.path.join(WORKING_DIR, "logs")

# 确保工作目录和日志目录存在
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志系统
log_manager = configure_logging(log_dir=LOG_DIR, default_level="info")
log_manager.log_system_info()
log_manager.log_config(CONFIG)

# 获取日志器
logger = get_logger("grag")


async def setup_rag() -> LightRAG:
    """
    设置并初始化RAG系统
    
    Returns:
        初始化好的LightRAG实例
    """
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


async def index_documents(rag: LightRAG, file_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    索引文档到RAG系统
    
    Args:
        rag: LightRAG实例
        file_paths: 文档文件路径列表
        
    Returns:
        问答对字典，按文档ID分组
    """
    # 获取文档处理配置
    doc_config = CONFIG["document"]
    # 创建进度条
    total_files = len(file_paths)
    progress_bar = create_progress_bar(total_files, "文档索引进度")
    
    # 存储问答对
    qa_pairs_by_doc = {}
    
    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        progress_bar.update(i, f"处理 {file_name}")
        logger.info(f"正在索引文档: {file_path}")
        
        # 根据文件类型读取内容
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 索引文档
            doc_id = await rag.index_document(
                content, 
                chunk_size=doc_config["chunk_size"], 
                chunk_overlap=doc_config["chunk_overlap"]
            )
            logger.info(f"文档已索引，ID: {doc_id}")
            qa_pairs_by_doc[doc_id] = []
            
        elif file_path.endswith('.docx'):
            # 使用unstructured库处理docx文件
            try:
                from unstructured.partition.docx import partition_docx
                elements = partition_docx(file_path)
                content = "\n".join([str(e) for e in elements])
                # 索引文档
                doc_id = await rag.index_document(
                    content, 
                    chunk_size=doc_config["chunk_size"], 
                    chunk_overlap=doc_config["chunk_overlap"]
                )
                logger.info(f"文档已索引，ID: {doc_id}")
                qa_pairs_by_doc[doc_id] = []
            except ImportError:
                logger.error("请安装unstructured库以处理docx文件: pip install unstructured")
                progress_bar.update(i+1, f"跳过 {file_name} (缺少unstructured库)")
                continue
                
        elif file_path.endswith('.json'):
            # 处理HotpotQA数据集
            try:
                doc_id, qa_pairs = await process_hotpotqa_dataset(rag, file_path)
                if doc_id:
                    logger.info(f"HotpotQA数据集已索引，ID: {doc_id}，问答对数量: {len(qa_pairs)}")
                    qa_pairs_by_doc[doc_id] = qa_pairs
                else:
                    logger.warning(f"处理HotpotQA数据集失败: {file_path}")
            except Exception as e:
                logger.error(f"处理JSON文件时出错: {e}")
                progress_bar.update(i+1, f"跳过 {file_name} (处理错误)")
                continue
        else:
            logger.warning(f"不支持的文件类型: {file_path}")
            progress_bar.update(i+1, f"跳过 {file_name} (不支持的文件类型)")
            continue
        
        progress_bar.update(i+1, f"完成 {file_name}")
    
    # 完成进度条
    progress_bar.finish()
    return qa_pairs_by_doc


async def query_and_process(
    rag: LightRAG, 
    question: str, 
    mode: str = None
) -> Tuple[str, List[str]]:
    # 如果未指定模式，使用配置中的默认模式
    if mode is None:
        mode = CONFIG["query"]["default_mode"]
    """
    查询RAG系统并处理结果
    
    Args:
        rag: LightRAG实例
        question: 查询问题
        mode: 查询模式 (local, global, hybrid, naive)
        
    Returns:
        回答和来源列表
    """
    # 设置查询参数
    param = QueryParam(
        mode=mode,
        top_k=CONFIG["query"]["top_k"],
        max_token_for_text_unit=CONFIG["query"]["max_token_for_text_unit"],
        max_token_for_global_context=CONFIG["query"]["max_token_for_global_context"],
        max_token_for_local_context=CONFIG["query"]["max_token_for_local_context"]
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


async def scan_and_index_data_directory(rag: LightRAG, data_dir: str = "data") -> Dict[str, List[Dict[str, Any]]]:
    """
    扫描数据目录并索引所有支持的文件
    
    Args:
        rag: LightRAG实例
        data_dir: 数据目录路径
        
    Returns:
        问答对字典，按文档ID分组
    """
    logger.info(f"开始扫描数据目录: {data_dir}")
    
    # 扫描数据目录
    file_types = scan_data_directory(data_dir)
    
    # 合并所有文件路径
    all_files = []
    for file_type, files in file_types.items():
        if file_type != "other":  # 跳过不支持的文件类型
            all_files.extend(files)
    
    if not all_files:
        logger.warning(f"数据目录 {data_dir} 中没有找到支持的文件")
        return {}
    
    # 索引所有文件
    logger.info(f"开始索引 {len(all_files)} 个文件")
    qa_pairs_by_doc = await index_documents(rag, all_files)
    
    return qa_pairs_by_doc


async def main():
    """主函数 - 已弃用，保留仅作参考"""
    # 此函数已被run_interactive()替代
    logger.warning("main()函数已弃用，请使用新的交互式界面")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Demo")
    parser.add_argument('--mode', type=str, default=CONFIG["query"]["default_mode"], 
                        choices=['local', 'global', 'hybrid', 'naive'], help='选择检索模式')
    parser.add_argument('--demo', action='store_true', help='运行演示问题')
    args = parser.parse_args()

    async def run_interactive():
        # 设置RAG系统
        rag = await setup_rag()
        
        # 自动扫描并索引数据目录中的所有支持文件
        data_folder = './data'
        qa_pairs = await scan_and_index_data_directory(rag, data_folder)
        
        # 保存问答对到文件，用于后续评估
        if qa_pairs:
            qa_file_path = os.path.join(WORKING_DIR, "qa_pairs.json")
            with open(qa_file_path, "w", encoding="utf-8") as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            logger.info(f"问答对已保存到 {qa_file_path}")
            print(f"问答对已保存到 {qa_file_path}")
        
        # 如果指定了demo参数，运行演示问题
        if args.demo:
            # 示例查询
            questions = [
                "公明镇镇长有谁担任过?"
            ]
            
            for question in questions:
                print(f"\n问题: {question}")
                
                # 使用不同模式查询
                for mode in ["hybrid"]:
                    print(f"\n模式: {mode}")
                    answer, sources = await query_and_process(rag, question, mode)
                    print(f"回答: {answer}")
                    
                    if sources:
                        print("\n来源:")
                        for source in sources:
                            print(f"- {source}")
        
        # 交互式查询
        # while True:
        #     try:
        #         query = input("\n问题 (输入'exit'退出): ")
        #         if query.lower() in ['exit', 'quit', '退出']:
        #             break
                
        #         print(f"模式: {args.mode}")
        #         param = QueryParam(mode=args.mode)
        #         # 只调用用户指定模式
        #         response = await rag.aquery(query, param=param)
        #         print("回答:", response)
        #     except KeyboardInterrupt:
        #         print("\n退出程序")
        #         break
        #     except Exception as e:
        #         print(f"发生错误: {e}")
    
    # 运行交互式查询
    asyncio.run(run_interactive())
