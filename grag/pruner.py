import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import logging

from grag.utils.common import get_logger

logger = get_logger("grag.pruner")

class FocalLoss(nn.Module):
    """
    Focal Loss实现，用于处理类别不平衡问题
    
    相比BCE损失，Focal Loss对难分类样本给予更高权重，对易分类样本降低权重
    适合处理标签噪声较大和类别不平衡的数据
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 正样本权重，用于平衡正负样本，默认为0.25
            gamma: 聚焦参数，值越大对难分类样本关注越多，默认为2.0
            reduction: 损失计算方式，可选'none'|'mean'|'sum'，默认为'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: 预测logits，形状为 [batch_size, 1]
            targets: 目标标签，形状为 [batch_size, 1]
            
        Returns:
            计算得到的损失值
        """
        # 计算BCE损失
        bce_loss = self.bce_with_logits(inputs, targets)
        
        # 计算概率
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # 计算alpha权重
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # 计算Focal Loss
        focal_loss = alpha_weight * (1 - pt) ** self.gamma * bce_loss
        
        # 根据reduction方式返回损失
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        else:  # 'sum'
            return focal_loss.sum()

class PrunerMLP(nn.Module):
    """
    用于图剪枝的MLP模型
    
    输入为拼接的节点嵌入和结构特征，输出为边的保留概率
    简化模型结构以减轻过拟合，使用较高的dropout率
    """
    def __init__(self, input_dim: int = 258, hidden_dim: int = 128, dropout_rate: float = 0.4):
        """
        初始化剪枝模型
        
        Args:
            input_dim: 输入维度，默认为258（两个128维节点嵌入的拼接 + 2个结构特征）
            hidden_dim: 隐藏层维度，默认为128
            dropout_rate: Dropout比率，默认为0.4（提高以减轻过拟合）
        """
        super(PrunerMLP, self).__init__()
        
        # 简化模型结构，减少层数以减轻过拟合
        # 第一个隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 保留批归一化
        self.dropout1 = nn.Dropout(dropout_rate)  # 提高dropout率
        
        # 第二个隐藏层 - 减小维度
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)  # 保留批归一化
        self.dropout2 = nn.Dropout(dropout_rate)  # 提高dropout率
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, input_dim]
            
        Returns:
            边的保留概率的logits，形状为 [batch_size, 1]
        """
        # 第一层 - 使用LeakyReLU代替ReLU以减轻过拟合
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout1(x)
        
        # 第二层 - 同样使用LeakyReLU
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout2(x)
        
        # 输出层
        x = self.fc_out(x)
        return x

class GraphPruner:
    """
    图剪枝器，用于优化知识图谱结构，减少冗余边
    
    使用轻量级MLP模型学习边的保留概率，支持混合精度训练和学习率自动调整
    """
    def __init__(self, 
                 embedding_dim: int = 128, 
                 hidden_dim: int = 128,
                 dropout_rate: float = 0.4,  # 提高dropout率以减轻过拟合
                 device: str = None):
        """
        初始化图剪枝器
        
        Args:
            embedding_dim: 单个节点的嵌入维度，默认为128
            hidden_dim: 模型隐藏层维度，默认为128
            dropout_rate: Dropout比率，用于减轻过拟合，默认为0.4
            device: 运行设备，默认为None（自动选择）
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.input_dim = embedding_dim * 2 + 12  # 两个节点嵌入拼接 + 12个结构特征
        self.best_model_state = None  # 用于保存最佳模型状态
        self.best_val_loss = float('inf')  # 用于早停机制
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = PrunerMLP(self.input_dim, hidden_dim, dropout_rate).to(self.device)
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型参数量: {total_params}")
        
    def _prepare_edge_features(self, 
                              node_embeddings: Dict[str, np.ndarray], 
                              edges: List[Tuple[str, str]],
                              graph_structure: Dict[str, List[str]] = None) -> torch.Tensor:
        """
        准备边特征（拼接的节点嵌入和结构特征）
        
        Args:
            node_embeddings: 节点ID到嵌入的映射
            edges: 边列表，每条边为源节点ID和目标节点ID的元组
            graph_structure: 图结构，节点ID到其邻居节点ID列表的映射
            
        Returns:
            边特征张量，包含节点嵌入和结构特征
        """
        # 如果没有提供图结构，则创建一个空的图结构
        if graph_structure is None:
            graph_structure = {}
            # 从边列表构建图结构
            for src_id, tgt_id in edges:
                if src_id not in graph_structure:
                    graph_structure[src_id] = []
                if tgt_id not in graph_structure:
                    graph_structure[tgt_id] = []
                graph_structure[src_id].append(tgt_id)
                graph_structure[tgt_id].append(src_id)  # 假设是无向图
        
        # 预计算节点度数 - 用于后续特征计算
        node_degrees = {node: len(neighbors) for node, neighbors in graph_structure.items()}
        
        # 计算全局图统计信息
        avg_degree = np.mean([len(neighbors) for neighbors in graph_structure.values()]) if graph_structure else 0
        max_degree = max([len(neighbors) for neighbors in graph_structure.values()]) if graph_structure else 0
        
        edge_features = []
        for src_id, tgt_id in edges:
            if src_id not in node_embeddings or tgt_id not in node_embeddings:
                logger.warning(f"节点 {src_id} 或 {tgt_id} 不在嵌入字典中")
                continue
            
            # 获取节点的邻居集合
            src_neighbors = set(graph_structure.get(src_id, []))
            tgt_neighbors = set(graph_structure.get(tgt_id, []))
            
            # 基础结构特征
            # 1. 共同邻居数量
            common_neighbors = len(src_neighbors.intersection(tgt_neighbors))
            
            # 2. Jaccard相似度 = |A∩B| / |A∪B|
            union_size = len(src_neighbors.union(tgt_neighbors))
            jaccard_similarity = common_neighbors / max(union_size, 1)  # 避免除零错误
            
            # 增强结构特征
            # 3. 节点度数特征
            src_degree = node_degrees.get(src_id, 0)
            tgt_degree = node_degrees.get(tgt_id, 0)
            degree_product = src_degree * tgt_degree  # 偏好依附度量
            degree_sum = src_degree + tgt_degree
            degree_diff = abs(src_degree - tgt_degree)
            
            # 4. 度数相对于图平均度数的比率
            src_degree_ratio = src_degree / max(avg_degree, 1)
            tgt_degree_ratio = tgt_degree / max(avg_degree, 1)
            
            # 5. Adamic-Adar指数 - 共同邻居的权重和，权重为邻居度数的倒数
            adamic_adar = 0
            for common_neighbor in src_neighbors.intersection(tgt_neighbors):
                neighbor_degree = node_degrees.get(common_neighbor, 0)
                if neighbor_degree > 1:  # 避免除零
                    adamic_adar += 1.0 / np.log(neighbor_degree)
            
            # 6. 资源分配指数 - 类似Adamic-Adar，但使用度数的倒数
            resource_allocation = 0
            for common_neighbor in src_neighbors.intersection(tgt_neighbors):
                neighbor_degree = node_degrees.get(common_neighbor, 0)
                if neighbor_degree > 0:  # 避免除零
                    resource_allocation += 1.0 / neighbor_degree
            
            # 7. 共同邻居比例 - 共同邻居数量除以可能的最大共同邻居数量
            max_possible_common = min(src_degree, tgt_degree)
            common_neighbor_ratio = common_neighbors / max(max_possible_common, 1)
            
            # 拼接源节点和目标节点的嵌入
            src_emb = node_embeddings[src_id]
            tgt_emb = node_embeddings[tgt_id]
            
            # 将节点嵌入和所有结构特征拼接在一起
            structural_features = [
                common_neighbors, 
                jaccard_similarity,
                src_degree,
                tgt_degree,
                degree_product,
                degree_sum,
                degree_diff,
                src_degree_ratio,
                tgt_degree_ratio,
                adamic_adar,
                resource_allocation,
                common_neighbor_ratio
            ]
            
            edge_emb = np.concatenate([src_emb, tgt_emb, structural_features])
            edge_features.append(edge_emb)
            
        return torch.tensor(np.array(edge_features), dtype=torch.float32)
    
    def train(self, 
              node_embeddings: Dict[str, np.ndarray], 
              edges: List[Tuple[str, str]], 
              edge_labels: List[int],
              graph_structure: Dict[str, List[str]] = None,
              batch_size: int = 128,
              epochs: int = 100,  # 增加训练轮数
              lr: float = 5e-4,  # 适当的初始学习率
              weight_decay: float = 1e-3,  # 权重衰减以加强正则化
              use_amp: bool = True,
              k_folds: int = 5,  # K折交叉验证的折数
              focal_alpha: float = 0.25,  # Focal Loss的alpha参数
              focal_gamma: float = 2.0) -> Dict[str, List[float]]:  # 使用K折交叉验证替代单一验证集
        """
        训练图剪枝模型
        
        Args:
            node_embeddings: 节点ID到嵌入的映射
            edges: 边列表，每条边为源节点ID和目标节点ID的元组
            edge_labels: 边标签列表，1表示保留，0表示剪枝
            graph_structure: 图结构，节点ID到其邻居节点ID列表的映射，默认为None
            batch_size: 批次大小，默认为128
            epochs: 训练轮数，默认为100
            lr: 学习率，默认为5e-4
            weight_decay: 权重衰减，默认为1e-3，用于L2正则化
            use_amp: 是否使用自动混合精度训练，默认为True
            k_folds: K折交叉验证的折数，默认为5
            focal_alpha: Focal Loss的alpha参数，默认为0.25
            focal_gamma: Focal Loss的gamma参数，默认为2.0
            
        Returns:
            包含训练历史的字典
        """
        # 准备数据
        edge_features = self._prepare_edge_features(node_embeddings, edges, graph_structure)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32).view(-1, 1)
        
        # 数据增强 - 添加少量高斯噪声到训练特征
        def add_noise(features, noise_level=0.05):
            noise = torch.randn_like(features) * noise_level
            return features + noise
        
        # 检查类别平衡
        pos_count = edge_labels.sum().item()
        neg_count = len(edge_labels) - pos_count
        pos_ratio = pos_count / len(edge_labels)
        logger.info(f"数据集正样本: {pos_count}, 负样本: {neg_count}, 比例: {pos_ratio:.2f}")
        
        # 设置混合精度训练
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # 训练历史
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        
        # 用于保存最佳模型
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # 实现K折交叉验证
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # 记录每折的验证损失
        fold_val_losses = []
        
        logger.info(f"开始{k_folds}折交叉验证训练...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(edge_features)):
            logger.info(f"开始第{fold+1}/{k_folds}折训练")
            
            # 重置模型参数
            self.model = PrunerMLP(self.input_dim, self.hidden_dim, self.dropout_rate).to(self.device)
            
            # 准备当前折的数据
            train_features = add_noise(edge_features[train_idx]).to(self.device)
            train_labels = edge_labels[train_idx].to(self.device)
            val_features = edge_features[val_idx].to(self.device)
            val_labels = edge_labels[val_idx].to(self.device)
            
            # 记录当前折数据集大小
            logger.info(f"第{fold+1}折 - 训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
            
            # 检查当前折类别平衡
            train_pos = train_labels.sum().item()
            train_neg = len(train_labels) - train_pos
            logger.info(f"第{fold+1}折 - 训练集正样本: {train_pos}, 负样本: {train_neg}, 比例: {train_pos/len(train_labels):.2f}")
            
            val_pos = val_labels.sum().item()
            val_neg = len(val_labels) - val_pos
            logger.info(f"第{fold+1}折 - 验证集正样本: {val_pos}, 负样本: {val_neg}, 比例: {val_pos/len(val_labels):.2f}")
            
            # 创建数据加载器
            train_dataset = TensorDataset(train_features, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # 设置优化器
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # 设置损失函数 - 使用Focal Loss处理类别不平衡和标签噪声
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"使用Focal Loss，alpha={focal_alpha}, gamma={focal_gamma}")
            
            # 使用ReduceLROnPlateau调度器，根据验证损失自动调整学习率
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, 
                patience=5, min_lr=lr/20, verbose=True
            )
            
            # 当前折的最佳验证损失
            fold_best_val_loss = float('inf')
            fold_best_model_state = None
            
            # 训练循环
            for epoch in range(epochs):
                # 训练模式
                self.model.train()
                train_loss = 0.0
                
                for batch_features, batch_labels in train_loader:
                    optimizer.zero_grad()
                    
                    # 使用自动混合精度
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = self.model(batch_features)
                        loss = criterion(outputs, batch_labels)
                    
                    # 反向传播
                    scaler.scale(loss).backward()
                    
                    # 添加梯度裁剪，防止梯度爆炸
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                
                # 计算平均训练损失
                train_loss /= len(train_loader)
                
                # 验证模式
                self.model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        val_outputs = self.model(val_features)
                        val_loss = criterion(val_outputs, val_labels).item()
                        
                        # 计算准确率
                        val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
                        val_accuracy = (val_preds == val_labels).float().mean().item()
                
                # 更新学习率调度器 - 根据验证损失调整学习率
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # 保存当前折的最佳模型
                if val_loss < fold_best_val_loss:
                    fold_best_val_loss = val_loss
                    fold_best_model_state = self.model.state_dict().copy()
                    logger.info(f"第{fold+1}折 - 保存新的最佳模型，验证损失: {fold_best_val_loss:.4f}")
                
                # 每5个epoch记录一次日志，减少输出
                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                    logger.info(f"第{fold+1}折 - Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {train_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}, "
                              f"Val Accuracy: {val_accuracy:.4f}, "
                              f"LR: {current_lr:.6f}")
            
            # 记录当前折的最佳验证损失
            fold_val_losses.append(fold_best_val_loss)
            logger.info(f"第{fold+1}折训练完成，最佳验证损失: {fold_best_val_loss:.4f}")
            
            # 如果当前折的模型是全局最佳，则保存
            if fold_best_val_loss < self.best_val_loss:
                self.best_val_loss = fold_best_val_loss
                self.best_model_state = fold_best_model_state
                logger.info(f"更新全局最佳模型，来自第{fold+1}折，验证损失: {self.best_val_loss:.4f}")
        
        # 计算平均验证损失
        avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)
        logger.info(f"K折交叉验证完成，平均验证损失: {avg_val_loss:.4f}，最佳验证损失: {self.best_val_loss:.4f}")
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"已恢复全局最佳模型，验证损失: {self.best_val_loss:.4f}")
        
        return history
    
    def prune(self, 
              node_embeddings: Dict[str, np.ndarray], 
              edges: List[Tuple[str, str]],
              graph_structure: Dict[str, List[str]] = None,
              threshold: float = 0.5,
              top_k_ratio: float = None,
              min_edges_to_keep: int = None,
              pruning_strategy: str = 'top_k') -> List[Tuple[str, str, float]]:
        """
        对图进行剪枝
        
        Args:
            node_embeddings: 节点ID到嵌入的映射
            edges: 边列表，每条边为源节点ID和目标节点ID的元组
            graph_structure: 图结构，节点ID到其邻居节点ID列表的映射，默认为None
            threshold: 保留边的概率阈值，默认为0.5，仅在pruning_strategy='threshold'时使用
            top_k_ratio: Top-K策略下保留的边比例，默认为None，若设置则覆盖threshold
            min_edges_to_keep: 至少保留的边数量，默认为None
            pruning_strategy: 剪枝策略，可选'threshold'或'top_k'，默认为'top_k'
            
        Returns:
            保留的边列表，每条边为(源节点ID, 目标节点ID, 保留概率)的元组
        """
        # 准备数据
        edge_features = self._prepare_edge_features(node_embeddings, edges, graph_structure)
        edge_features = edge_features.to(self.device)
        
        # 评估模式
        self.model.eval()
        
        # 使用自动混合精度
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # 应用sigmoid函数获取概率值
                logits = self.model(edge_features)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        # 创建边和概率的索引列表
        edge_prob_pairs = [(i, float(probs[i])) for i in range(len(probs)) if i < len(edges)]
        
        # 根据剪枝策略选择保留的边
        if pruning_strategy == 'top_k':
            # 按概率从高到低排序
            edge_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 确定要保留的边数量
            if top_k_ratio is not None:
                k = max(int(len(edge_prob_pairs) * top_k_ratio), 1)  # 至少保留1条边
            else:
                # 如果未指定top_k_ratio，则使用threshold作为默认比例
                k = sum(1 for _, prob in edge_prob_pairs if prob >= threshold)
            
            # 如果指定了最小保留边数，确保至少保留这么多边
            if min_edges_to_keep is not None:
                k = max(k, min_edges_to_keep)
            
            # 取前k个边
            selected_indices = [idx for idx, _ in edge_prob_pairs[:k]]
        else:  # 'threshold'
            # 使用阈值筛选
            selected_indices = [idx for idx, prob in edge_prob_pairs if prob >= threshold]
            
            # 如果指定了最小保留边数，但筛选后的边数不足，则补充边
            if min_edges_to_keep is not None and len(selected_indices) < min_edges_to_keep:
                # 对未选中的边按概率排序
                remaining_edges = [(idx, prob) for idx, prob in edge_prob_pairs if idx not in selected_indices]
                remaining_edges.sort(key=lambda x: x[1], reverse=True)
                
                # 补充到达到最小边数
                additional_edges = remaining_edges[:min_edges_to_keep - len(selected_indices)]
                selected_indices.extend([idx for idx, _ in additional_edges])
        
        # 构建最终保留的边列表
        pruned_edges = []
        for i in selected_indices:
            src_id, tgt_id = edges[i]
            pruned_edges.append((src_id, tgt_id, float(probs[i])))
        
        logger.info(f"原始边数: {len(edges)}, 剪枝后边数: {len(pruned_edges)}")
        logger.info(f"使用剪枝策略: {pruning_strategy}, 保留比例: {len(pruned_edges)/max(len(edges), 1):.2f}")
        
        return pruned_edges
    
    def save_model(self, model_path: str):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型参数和配置
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "best_val_loss": self.best_val_loss
        }, model_path)
        
        logger.info(f"模型已保存至 {model_path}")
        logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")
    
    def load_model(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型加载路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        
        # 加载模型参数和配置
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 更新配置
        self.embedding_dim = checkpoint.get("embedding_dim", self.embedding_dim)
        self.hidden_dim = checkpoint.get("hidden_dim", self.hidden_dim)
        self.dropout_rate = checkpoint.get("dropout_rate", self.dropout_rate)
        self.input_dim = self.embedding_dim * 2
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        # 重新初始化模型
        self.model = PrunerMLP(self.input_dim, self.hidden_dim, self.dropout_rate).to(self.device)
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"模型已从 {model_path} 加载")
        logger.info(f"加载的模型验证损失: {self.best_val_loss:.4f}")