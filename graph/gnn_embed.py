# graph/gnn_embed.py

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from tqdm import tqdm


class GNNEncoder(nn.Module):
    """
    图神经网络编码器，用于生成节点的结构感知嵌入
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 model_type: str = "GAT", num_layers: int = 2, heads: int = 4, dropout: float = 0.2):
        """
        初始化GNN编码器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            model_type: GNN模型类型，可选GAT、GraphSAGE、GCN
            num_layers: GNN层数
            heads: 注意力头数（仅用于GAT）
            dropout: Dropout概率
        """
        super(GNNEncoder, self).__init__()
        
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 选择GNN层类型
        if model_type == "GAT":
            # GAT层
            self.convs = nn.ModuleList()
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads))
            
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
            
            self.convs.append(GATConv(hidden_dim * heads, output_dim // heads, heads=heads, concat=True))
            
        elif model_type == "GraphSAGE":
            # GraphSAGE层
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.convs.append(SAGEConv(hidden_dim, output_dim))
            
        elif model_type == "GCN":
            # GCN层
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.convs.append(GCNConv(hidden_dim, output_dim))
            
        else:
            raise ValueError(f"Unsupported GNN model type: {model_type}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征矩阵，形状为[num_nodes, input_dim]
            edge_index: 边索引，形状为[2, num_edges]
            
        Returns:
            更新后的节点特征矩阵，形状为[num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # 非最后一层
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GraphEmbedder:
    """
    图嵌入器，负责生成图结构感知的节点嵌入
    """
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        初始化图嵌入器
        
        Args:
            config_path: 配置文件路径
        """
        # 处理配置文件路径
        if not os.path.isabs(config_path):
            # 如果是相对路径，则相对于当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, config_path)
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")
        
        # GNN参数
        try:
            gnn_config = self.config.get('gnn', {})
            self.model_type = gnn_config.get('model_type', 'GAT')
            self.hidden_dim = gnn_config.get('hidden_dim', 256)
            self.output_dim = gnn_config.get('output_dim', 768)
            self.num_layers = gnn_config.get('num_layers', 2)
            self.heads = gnn_config.get('heads', 4)
            self.dropout = gnn_config.get('dropout', 0.2)
        except Exception as e:
            print(f"Warning: Error loading GNN parameters from config: {e}. Using default values.")
            # 使用默认值
            self.model_type = 'GAT'
            self.hidden_dim = 256
            self.output_dim = 768
            self.num_layers = 2
            self.heads = 4
            self.dropout = 0.2
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型将在训练时初始化
        self.model = None
    
    def prepare_data(self, graph: nx.Graph, embeddings: Dict[str, np.ndarray]) -> Data:
        """
        准备PyTorch Geometric数据对象
        
        Args:
            graph: NetworkX图对象
            embeddings: 节点ID到嵌入向量的映射
            
        Returns:
            PyTorch Geometric数据对象
        """
        # 节点ID到索引的映射
        node_ids = list(graph.nodes())
        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # 节点特征
        x = np.stack([embeddings[node_id] for node_id in node_ids])
        x = torch.FloatTensor(x)
        
        # 边索引
        edge_index = []
        for src, dst in graph.edges():
            edge_index.append([node_id_to_idx[src], node_id_to_idx[dst]])
            edge_index.append([node_id_to_idx[dst], node_id_to_idx[src]])  # 无向图，添加反向边
        
        edge_index = torch.LongTensor(edge_index).t()  # 转置为[2, num_edges]形状
        
        # 边权重（可选）
        edge_weight = []
        for src, dst in graph.edges():
            weight = graph[src][dst].get('weight', 1.0)
            edge_weight.append(weight)
            edge_weight.append(weight)  # 无向图，添加反向边权重
        
        edge_weight = torch.FloatTensor(edge_weight)
        
        # 创建数据对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        
        # 将数据转移到设备前保存映射信息
        node_id_mapping = {
            'node_id_to_idx': node_id_to_idx,
            'node_ids': node_ids
        }
        
        # 将数据转移到设备
        # 将设备类型转换为字符串
        device_str = str(self.device)
        data = data.to(device_str)
        
        # 在转移后重新添加映射信息（作为普通属性，不作为张量）
        data.node_id_mapping = node_id_mapping
        
        return data
    
    def train(self, graph: nx.Graph, embeddings: Dict[str, np.ndarray], 
              num_epochs: int = 100, lr: float = 0.001) -> Dict[str, np.ndarray]:
        """
        训练GNN模型并生成结构感知的节点嵌入
        
        Args:
            graph: NetworkX图对象
            embeddings: 节点ID到嵌入向量的映射
            num_epochs: 训练轮数
            lr: 学习率
            
        Returns:
            节点ID到结构感知嵌入向量的映射
        """
        # 准备数据
        data = self.prepare_data(graph, embeddings)
        
        # 初始化模型
        # 获取输入维度，处理data.x为None的情况
        input_dim = data.x.shape[1] if data.x is not None else self.config.get('embedding', {}).get('dimension', 768)
        self.model = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            model_type=self.model_type,
            num_layers=self.num_layers,
            heads=self.heads,
            dropout=self.dropout
        ).to(self.device)
        
        # 定义优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 训练模型
        self.model.train()
        for epoch in tqdm(range(num_epochs), desc="Training GNN"):
            optimizer.zero_grad()
            
            # 前向传播
            out = self.model(data.x, data.edge_index)
            
            # 计算损失（使用自监督对比损失）
            # 确保edge_index不为None
            if data.edge_index is None:
                raise ValueError("edge_index cannot be None for contrastive loss calculation")
            loss = self.contrastive_loss(out, data.edge_index.to(self.device))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        # 生成嵌入
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(data.x, data.edge_index).cpu().numpy()
        
        # 构建节点ID到嵌入的映射
        id_to_embedding = {}
        node_ids = data.node_id_mapping['node_ids']  # 修改这里
        for i, node_id in enumerate(node_ids):
            id_to_embedding[node_id] = node_embeddings[i]
        
        return id_to_embedding
    
    def contrastive_loss(self, z: torch.Tensor, edge_index: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            z: 节点嵌入，形状为[num_nodes, dim]
            edge_index: 边索引，形状为[2, num_edges]
            tau: 温度参数
            
        Returns:
            对比损失
        """
        # 归一化嵌入
        z_norm = F.normalize(z, p=2, dim=1)
        
        # 计算所有节点对的相似度
        sim = torch.mm(z_norm, z_norm.t()) / tau
        
        # 创建正样本掩码（连接的节点对）
        pos_mask = torch.zeros_like(sim, dtype=torch.bool)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            pos_mask[src, dst] = True
        
        # 对角线设为False（自身不是正样本）
        pos_mask.fill_diagonal_(False)
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -log_prob[pos_mask].mean()
        
        return loss
    
    def save_model(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        加载模型
        
        Args:
            load_path: 加载路径
        """
        # 检查模型文件是否存在
        if not os.path.exists(load_path):
            raise ValueError(f"Model file not found: {load_path}")
        
        # 加载模型参数
        try:
            state_dict = torch.load(load_path, map_location=self.device)
        except Exception as e:
            print(f"Error loading model file: {e}")
            raise ValueError(f"Failed to load model from {load_path}: {e}")
        
        # 从配置中获取输入维度
        try:
            input_dim = self.config.get('embedding', {}).get('dimension', None)
        except (KeyError, TypeError):
            input_dim = None
        
        # 如果无法从配置中获取维度信息，尝试从state_dict推断
        if not input_dim:
            # 查找包含输入层权重的键
            for key in state_dict.keys():
                if 'convs.0' in key and 'weight' in key:
                    if self.model_type == 'GAT':
                        # GAT的第一层权重形状可能是[out_channels, in_channels]
                        input_dim = state_dict[key].size(1)
                    else:
                        # GCN和GraphSAGE的权重形状
                        input_dim = state_dict[key].size(1)
                    break
            
            # 如果仍然无法确定输入维度，使用默认值
            if not input_dim:
                print("Warning: Could not determine input dimension from model state. Using default value.")
                input_dim = 768  # 默认维度
        
        # 初始化模型
        self.model = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            model_type=self.model_type,
            num_layers=self.num_layers,
            heads=self.heads,
            dropout=self.dropout
        ).to(self.device)
        
        # 尝试加载参数
        try:
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model loaded from {load_path}")
        except Exception as e:
            print(f"Error details: {e}")
            print(f"Model structure: {self.model}")
            print(f"Expected state_dict keys: {list(self.model.state_dict().keys())[:5]}...")
            print(f"Provided state_dict keys: {list(state_dict.keys())[:5]}...")
            raise ValueError(f"Error loading model parameters: {e}")


    
    def generate_embeddings(self, graph: nx.Graph, embeddings: Dict[str, np.ndarray], 
                           save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        生成结构感知的节点嵌入
        
        Args:
            graph: NetworkX图对象
            embeddings: 节点ID到嵌入向量的映射
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            节点ID到结构感知嵌入向量的映射
        """
        # 训练模型并生成嵌入
        graph_embeddings = self.train(graph, embeddings)
        
        # 保存嵌入
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(graph_embeddings, f)
            print(f"Graph embeddings saved to {save_path}")
        
        return graph_embeddings


if __name__ == "__main__":
    # 示例用法
    import json
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.embedding import TextEmbedding
    
    # 加载图
    with open("../data/document_graph.pkl", 'rb') as f:
        graph = pickle.load(f)
    
    # 加载基础嵌入
    with open("../data/embeddings/base_embeddings.pkl", 'rb') as f:
        base_embeddings = pickle.load(f)
    
    # 初始化图嵌入器
    embedder = GraphEmbedder("../config.yaml")
    
    # 生成图嵌入
    graph_embeddings = embedder.generate_embeddings(
        graph,
        base_embeddings,
        save_path="../data/embeddings/graph_embeddings.pkl"
    )
    
    # 保存模型
    embedder.save_model("../data/gnn_model.pt")
    
    print(f"Generated graph embeddings for {len(graph_embeddings)} nodes")
    print(f"Embedding dimension: {next(iter(graph_embeddings.values())).shape}")