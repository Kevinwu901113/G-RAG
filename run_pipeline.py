# run_pipeline.py

import argparse
import json
import os
import pickle
from typing import Dict, List, Any, Optional

import yaml
from tqdm import tqdm

# 导入自定义模块
from utils.text_splitter import DocumentSplitter
from utils.embedding import TextEmbedding
from graph.build_graph import DocumentGraphBuilder
from graph.gnn_embed import GraphEmbedder
from retrieval.faiss_index import FaissRetriever
from generator.generate_answer import get_generator
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class GraphLightRAG:
    """
    Graph-Enhanced LightRAG 主流程类
    融合图结构嵌入的结构化检索生成框架
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化 GraphLightRAG
        
        Args:
            config_path: 配置文件路径
        """
        # 将配置文件路径转换为绝对路径
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
            
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化组件
        self.splitter = DocumentSplitter(config_path)
        self.embedder = TextEmbedding(config_path)
        self.graph_builder = DocumentGraphBuilder(config_path)
        self.graph_embedder = GraphEmbedder(config_path)
        self.retriever = FaissRetriever(config_path)
        self.generator = get_generator(config_path)
        
        # 数据路径
        self.raw_docs_dir = self.config['data']['raw_docs_dir']
        self.chunks_file = self.config['data']['chunks_file']
        self.graph_file = self.config['graph']['save_path']
        self.base_embeddings_file = os.path.join(self.config['embedding']['cache_dir'], "base_embeddings.pkl")
        self.graph_embeddings_file = os.path.join(self.config['embedding']['cache_dir'], "graph_embeddings.pkl")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.chunks_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.graph_file), exist_ok=True)
        os.makedirs(self.config['embedding']['cache_dir'], exist_ok=True)
        os.makedirs(self.config['retrieval']['index_path'], exist_ok=True)
    
    def process_documents(self, force_reprocess: bool = False) -> List[Dict[str, Any]]:
        """
        处理文档：切分文本、构建图、生成嵌入
        
        Args:
            force_reprocess: 是否强制重新处理
            
        Returns:
            处理后的文档列表
        """
        # 1. 文档切分
        if not os.path.exists(self.chunks_file) or force_reprocess:
            print("Step 1: Splitting documents...")
            documents = self.splitter.process_directory(self.raw_docs_dir, self.chunks_file)
        else:
            print("Step 1: Loading existing document chunks...")
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        
        print(f"Total chunks: {len(documents)}")
        
        # 如果没有文档，提前返回
        if len(documents) == 0:
            print("警告: 没有找到文档。请确保 raw_docs_dir 目录中包含文本文件。")
            return documents
        
        # 2. 计算基础嵌入
        if not os.path.exists(self.base_embeddings_file) or force_reprocess:
            print("Step 2: Computing base embeddings...")
            base_embeddings = self.embedder.embed_documents(documents, self.base_embeddings_file)
        else:
            print("Step 2: Loading existing base embeddings...")
            with open(self.base_embeddings_file, 'rb') as f:
                base_embeddings = pickle.load(f)
        
        print(f"Base embeddings: {len(base_embeddings)}")
        
        # 3. 构建文档图
        if not os.path.exists(self.graph_file) or force_reprocess:
            print("Step 3: Building document graph...")
            graph = self.graph_builder.build_graph(self.chunks_file, self.graph_file)
        else:
            print("Step 3: Loading existing document graph...")
            with open(self.graph_file, 'rb') as f:
                graph = pickle.load(f)
        
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # 如果图为空，提前返回
        if graph.number_of_nodes() == 0:
            print("警告: 生成的图没有节点。无法继续处理。")
            return documents
        
        # 4. 生成图嵌入
        if not os.path.exists(self.graph_embeddings_file) or force_reprocess:
            print("Step 4: Generating graph embeddings...")
            graph_embeddings = self.graph_embedder.generate_embeddings(
                graph, base_embeddings, self.graph_embeddings_file)
        else:
            print("Step 4: Loading existing graph embeddings...")
            with open(self.graph_embeddings_file, 'rb') as f:
                graph_embeddings = pickle.load(f)
        
        print(f"Graph embeddings: {len(graph_embeddings)}")
        
        # 5. 构建检索索引
        print("Step 5: Building retrieval index...")
        self.retriever.build_index(documents, base_embeddings, graph_embeddings)
        
        return documents
    
    def answer_query(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        回答查询
        
        Args:
            query: 用户查询
            top_k: 检索的文档数量，如果为None则使用配置中的值
            
        Returns:
            包含答案和检索文档的字典
        """
        # 1. 检索相关文档
        print(f"Query: {query}")
        print("Retrieving relevant documents...")
        documents = self.retriever.retrieve(query, top_k)
        
        # 2. 生成答案
        print("Generating answer...")
        answer = self.generator.generate(query, documents)
        
        # 3. 返回结果
        return {
            "query": query,
            "answer": answer,
            "documents": documents
        }
    
    def run_interactive(self):
        """
        运行交互式问答
        """
        print("\nGraph-Enhanced LightRAG Interactive Mode")
        print("Type 'exit' to quit\n")
        
        while True:
            query = input("\nEnter your question: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            result = self.answer_query(query)
            
            print("\nAnswer:")
            print(result["answer"])
            
            print("\nRetrieved Documents:")
            for i, doc in enumerate(result["documents"]):
                print(f"Document {i+1} (Score: {doc['score']:.4f}):")
                print(f"Source: {doc['metadata']['source']}")
                print(f"Content: {doc['content'][:150]}...")
                print()
    
    def evaluate(self, test_data_path: str) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_data_path: 测试数据路径，JSON格式，包含query和answer字段
            
        Returns:
            评估指标
        """
        # 加载测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = []
        for item in tqdm(test_data, desc="Evaluating"):
            query = item["query"]
            gold_answer = item["answer"]
            
            # 回答查询
            result = self.answer_query(query)
            result["gold_answer"] = gold_answer
            
            results.append(result)
        
        # 计算评估指标
        # 这里只是一个简单的实现，实际应用中应该使用更复杂的评估指标
        metrics = self._compute_metrics(results)
        
        # 保存结果
        output_path = os.path.join(self.config['experiment']['log_dir'], "evaluation_results.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "metrics": metrics
            }, f, ensure_ascii=False, indent=2)
        
        return metrics
    
    def _compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            results: 评估结果列表
            
        Returns:
            评估指标
        """
        # 这里只是一个简单的实现，实际应用中应该使用更复杂的评估指标
        from rouge import Rouge
        
        rouge = Rouge()
        rouge_scores = []
        
        for result in results:
            try:
                # 计算ROUGE分数
                scores = rouge.get_scores(result["answer"], result["gold_answer"])
                rouge_scores.append(scores[0])
            except Exception as e:
                print(f"Error computing ROUGE: {e}")
        
        # 计算平均分数
        avg_rouge_1 = sum(score["rouge-1"]["f"] for score in rouge_scores) / len(rouge_scores)
        avg_rouge_2 = sum(score["rouge-2"]["f"] for score in rouge_scores) / len(rouge_scores)
        avg_rouge_l = sum(score["rouge-l"]["f"] for score in rouge_scores) / len(rouge_scores)
        
        return {
            "rouge-1": avg_rouge_1,
            "rouge-2": avg_rouge_2,
            "rouge-l": avg_rouge_l
        }


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="Graph-Enhanced LightRAG")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--reprocess", action="store_true", help="强制重新处理文档")
    parser.add_argument("--evaluate", action="store_true", help="评估模式")
    parser.add_argument("--test_data", type=str, help="测试数据路径")
    parser.add_argument("--query", type=str, help="单次查询")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    
    args = parser.parse_args()
    
    # 初始化 GraphLightRAG
    rag = GraphLightRAG(args.config)
    
    # 处理文档
    documents = rag.process_documents(args.reprocess)
    
    # 如果没有文档，提示用户并退出
    if len(documents) == 0:
        print("\n错误: 没有找到任何文档。请将文本文件放入配置的 raw_docs_dir 目录中，然后重试。")
        return
    
    # 根据模式运行
    if args.evaluate:
        if not args.test_data:
            print("Error: --test_data is required for evaluation mode")
            return
        
        metrics = rag.evaluate(args.test_data)
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    elif args.query:
        result = rag.answer_query(args.query)
        print("\nAnswer:")
        print(result["answer"])
    
    elif args.interactive:
        rag.run_interactive()
    
    else:
        print("No action specified. Use --query, --interactive, or --evaluate")


if __name__ == "__main__":
    main()