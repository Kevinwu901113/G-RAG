import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
import logging

# 导入项目的日志管理器
from ..utils.logger_manager import get_logger

# 创建日志器
logger = get_logger("query_classifier")

# 标签映射常量
LABEL_MAPS = {
    "query_strategy": {"noRAG": 0, "hybrid": 1},
    "precision_required": {"no": 0, "yes": 1}
}

# 反向标签映射（用于推理时将数字转回文本标签）
REVERSE_LABEL_MAPS = {
    "query_strategy": {0: "noRAG", 1: "hybrid"},
    "precision_required": {0: "no", 1: "yes"}
}

class QueryDataset(Dataset):
    """
    用于多标签文本分类的数据集类
    """
    def __init__(self, texts: List[str], labels: Optional[List[List[int]]] = None, tokenizer=None, max_length=128):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表，每个元素是一个包含两个标签的列表 [query_strategy, precision_required]
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 对文本进行编码
        encoding = self.tokenizer(text, truncation=True, padding='max_length', 
                                 max_length=self.max_length, return_tensors='pt')
        
        # 移除批次维度
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # 如果有标签，则添加到返回项中
        if self.labels is not None:
            item['labels'] = torch.FloatTensor(self.labels[idx])
            
        return item

class QueryClassifierModel(torch.nn.Module):
    """
    基于BERT的多标签分类模型
    """
    def __init__(self, model_name="bert-base-uncased", num_labels=2, local_files_only=False):
        """
        初始化模型
        
        Args:
            model_name: 预训练模型名称或本地模型路径
            num_labels: 标签数量
            local_files_only: 是否只使用本地文件
        """
        super(QueryClassifierModel, self).__init__()
        self.num_labels = num_labels
        
        # 检查环境变量中是否指定了模型路径
        env_model_path = os.environ.get("BERT_MODEL_PATH", "")
        
        # 尝试加载模型的顺序：
        # 1. 从环境变量指定的路径加载
        # 2. 从指定的模型名称/路径加载（先尝试本地，再尝试在线）
        # 3. 尝试备用模型列表
        # 4. 使用随机初始化的模型
        
        bert_model = None
        
        # 1. 尝试从环境变量指定的路径加载
        if env_model_path and os.path.exists(env_model_path):
            logger.info(f"尝试从环境变量指定的路径加载BERT模型: {env_model_path}")
            try:
                bert_model = BertModel.from_pretrained(env_model_path, local_files_only=True)
                logger.info(f"成功从环境变量指定的路径加载BERT模型")
            except Exception as e:
                logger.warning(f"从环境变量指定的路径加载BERT模型失败: {str(e)}")
        
        # 2. 尝试从指定的模型名称/路径加载
        if bert_model is None:
            logger.info(f"尝试加载BERT模型: {model_name}")
            try:
                # 先尝试本地加载
                if local_files_only:
                    bert_model = BertModel.from_pretrained(model_name, local_files_only=True)
                    logger.info(f"成功从本地加载BERT模型: {model_name}")
                else:
                    # 先尝试本地加载，失败后尝试在线加载
                    try:
                        bert_model = BertModel.from_pretrained(model_name, local_files_only=True)
                        logger.info(f"成功从本地加载BERT模型: {model_name}")
                    except Exception:
                        logger.info(f"尝试从在线加载BERT模型: {model_name}")
                        bert_model = BertModel.from_pretrained(model_name, local_files_only=False)
                        logger.info(f"成功从在线加载BERT模型: {model_name}")
            except Exception as e:
                logger.warning(f"加载BERT模型 {model_name} 失败: {str(e)}")
        
        # 3. 尝试备用模型列表
        if bert_model is None:
            backup_models = ["bert-base-chinese", "bert-base-multilingual-uncased", "distilbert-base-uncased"]
            for backup_model in backup_models:
                logger.info(f"尝试加载备用BERT模型: {backup_model}")
                try:
                    # 先尝试本地加载
                    try:
                        bert_model = BertModel.from_pretrained(backup_model, local_files_only=True)
                        logger.info(f"成功从本地加载备用BERT模型: {backup_model}")
                        break
                    except Exception:
                        # 再尝试在线加载
                        bert_model = BertModel.from_pretrained(backup_model, local_files_only=False)
                        logger.info(f"成功从在线加载备用BERT模型: {backup_model}")
                        break
                except Exception as e:
                    logger.warning(f"加载备用BERT模型 {backup_model} 失败: {str(e)}")
        
        # 4. 使用随机初始化的模型
        if bert_model is None:
            from transformers import BertConfig
            logger.warning("无法加载任何预训练模型，将使用随机初始化的模型")
            config = BertConfig(hidden_size=768, num_attention_heads=12, num_hidden_layers=6)
            bert_model = BertModel(config)
            logger.info("已创建随机初始化的BERT模型")
        
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
    
    @classmethod
    def from_pretrained(cls, model_path, num_labels=2):
        """
        从预训练模型加载
        
        Args:
            model_path: 模型路径
            num_labels: 标签数量
            
        Returns:
            加载好的模型
        """
        logger.info(f"从预训练模型加载: {model_path}")
        
        # 创建模型实例
        model = cls(model_name=model_path, num_labels=num_labels, local_files_only=True)
        
        # 尝试加载模型权重
        try:
            # 检查是否存在模型权重文件
            model_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                # 加载模型权重
                state_dict = torch.load(model_file, map_location="cpu")
                model.load_state_dict(state_dict)
                logger.info(f"成功加载模型权重: {model_file}")
            else:
                logger.warning(f"找不到模型权重文件: {model_file}")
        except Exception as e:
            logger.error(f"加载模型权重时出错: {str(e)}")
        
        return model
    
    def save_pretrained(self, output_dir):
        """
        保存模型到指定目录
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型权重
        model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), model_file)
        
        # 保存模型配置
        config_file = os.path.join(output_dir, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({
                "hidden_size": self.bert.config.hidden_size,
                "num_labels": self.num_labels
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存到: {output_dir}")
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            token_type_ids: 标记类型ID
            labels: 标签
            
        Returns:
            输出结果字典
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def load_data(data_path: str) -> Tuple[List[str], List[List[int]]]:
    """
    加载JSONL格式的训练数据
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        texts: 文本列表
        labels: 标签列表
    """
    texts = []
    labels = []
    
    logger.info(f"从 {data_path} 加载数据")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    query = data.get("query", "")
                    query_strategy = data.get("query_strategy", "noRAG")
                    precision_required = data.get("precision_required", "no")
                    
                    # 将文本标签转换为数字标签
                    query_strategy_label = LABEL_MAPS["query_strategy"].get(query_strategy, 0)
                    precision_required_label = LABEL_MAPS["precision_required"].get(precision_required, 0)
                    
                    texts.append(query)
                    labels.append([query_strategy_label, precision_required_label])
                except json.JSONDecodeError:
                    logger.warning(f"无法解析JSON行: {line}")
                    continue
    except Exception as e:
        logger.error(f"加载数据时出错: {str(e)}")
        raise
    
    logger.info(f"成功加载 {len(texts)} 条数据")
    return texts, labels

def train_model(data_path: str, output_dir: str = "./query_classifier_model", batch_size: int = 8, num_epochs: int = 5, model_name: str = "bert-base-uncased", local_model_path: str = None):
    """
    训练多标签分类模型
    
    Args:
        data_path: 训练数据路径
        output_dir: 模型输出目录
        batch_size: 批次大小
        num_epochs: 训练轮数
        model_name: 预训练模型名称
        local_model_path: 本地模型路径，如果提供则优先使用本地模型
    """
    # 加载数据
    texts, labels = load_data(data_path)
    
    # 初始化分词器
    tokenizer = None
    
    # 尝试加载分词器的顺序：
    # 1. 从指定的本地模型路径加载
    # 2. 从环境变量指定的本地模型路径加载
    # 3. 从预训练模型名称加载（先尝试本地，再尝试在线）
    # 4. 使用备用模型
    # 5. 使用简单分词器
    
    # 检查环境变量中是否指定了模型路径
    env_model_path = os.environ.get("BERT_MODEL_PATH", "")
    
    try:
        # 1. 优先尝试从指定的本地路径加载
        if local_model_path and os.path.exists(local_model_path):
            logger.info(f"从指定的本地路径加载分词器: {local_model_path}")
            try:
                tokenizer = BertTokenizer.from_pretrained(local_model_path, local_files_only=True)
                logger.info(f"成功从指定的本地路径加载分词器")
            except Exception as e:
                logger.warning(f"从指定的本地路径加载分词器失败: {str(e)}")
        
        # 2. 尝试从环境变量指定的路径加载
        if tokenizer is None and env_model_path and os.path.exists(env_model_path):
            logger.info(f"尝试从环境变量指定的路径加载分词器: {env_model_path}")
            try:
                tokenizer = BertTokenizer.from_pretrained(env_model_path, local_files_only=True)
                logger.info(f"成功从环境变量指定的路径加载分词器")
                local_model_path = env_model_path  # 更新本地模型路径
            except Exception as e:
                logger.warning(f"从环境变量指定的路径加载分词器失败: {str(e)}")
        
        # 3. 尝试从预训练模型名称加载（先尝试本地，再尝试在线）
        if tokenizer is None:
            logger.info(f"尝试从预训练模型加载分词器: {model_name}")
            try:
                # 先尝试本地加载
                tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
                logger.info(f"成功从本地加载预训练分词器: {model_name}")
            except Exception as e:
                logger.warning(f"从本地加载预训练分词器失败，尝试在线加载: {str(e)}")
                try:
                    # 再尝试在线加载
                    tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=False)
                    logger.info(f"成功从在线加载预训练分词器: {model_name}")
                except Exception as e2:
                    logger.warning(f"从在线加载预训练分词器失败: {str(e2)}")
        
        # 4. 尝试使用备用模型
        if tokenizer is None:
            backup_models = ["bert-base-chinese", "bert-base-multilingual-uncased", "distilbert-base-uncased"]
            for backup_model in backup_models:
                logger.info(f"尝试使用备用模型分词器: {backup_model}")
                try:
                    # 先尝试本地加载
                    tokenizer = BertTokenizer.from_pretrained(backup_model, local_files_only=True)
                    logger.info(f"成功从本地加载备用分词器: {backup_model}")
                    model_name = backup_model  # 更新模型名称
                    break
                except Exception:
                    try:
                        # 再尝试在线加载
                        tokenizer = BertTokenizer.from_pretrained(backup_model, local_files_only=False)
                        logger.info(f"成功从在线加载备用分词器: {backup_model}")
                        model_name = backup_model  # 更新模型名称
                        break
                    except Exception as e3:
                        logger.warning(f"加载备用分词器 {backup_model} 失败: {str(e3)}")
        
        # 5. 如果所有尝试都失败，使用简单分词器
        if tokenizer is None:
            from transformers import BasicTokenizer
            logger.warning("所有预训练分词器加载失败，使用简单分词器替代")
            
            # 创建一个简单的分词器包装类，模拟BertTokenizer的接口
            class SimpleTokenizer:
                def __init__(self):
                    self.basic_tokenizer = BasicTokenizer(do_lower_case=True)
                    self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
                    self.max_len = 512
                
                def __call__(self, text, truncation=True, padding='max_length', max_length=128, return_tensors=None):
                    # 简单分词
                    tokens = ["[CLS]"] + self.basic_tokenizer.tokenize(text)[:max_length-2] + ["[SEP]"]
                    
                    # 转换为ID
                    input_ids = []
                    for token in tokens:
                        if token in self.vocab:
                            input_ids.append(self.vocab[token])
                        else:
                            # 为新词分配ID
                            self.vocab[token] = len(self.vocab)
                            input_ids.append(self.vocab[token])
                    
                    # 填充
                    if padding == 'max_length':
                        attention_mask = [1] * len(input_ids)
                        padding_length = max_length - len(input_ids)
                        input_ids = input_ids + [0] * padding_length
                        attention_mask = attention_mask + [0] * padding_length
                    
                    # 转换为张量
                    if return_tensors == 'pt':
                        import torch
                        return {
                            "input_ids": torch.tensor([input_ids]),
                            "attention_mask": torch.tensor([attention_mask])
                        }
                    
                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    }
                
                def save_pretrained(self, path):
                    os.makedirs(path, exist_ok=True)
                    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
                        json.dump({"type": "simple_tokenizer"}, f)
            
            tokenizer = SimpleTokenizer()
            logger.info("已初始化简单分词器")
    
    except Exception as e:
        logger.error(f"所有分词器加载方法都失败: {str(e)}")
        raise RuntimeError("无法加载任何可用的分词器模型")
    
    # 创建数据集
    dataset = QueryDataset(texts, labels, tokenizer)
    
    # 初始化模型
    model = None
    
    try:
        # 优先尝试从本地路径加载
        if local_model_path and os.path.exists(local_model_path):
            logger.info(f"从本地路径加载模型: {local_model_path}")
            try:
                model = QueryClassifierModel(local_model_path, num_labels=2, local_files_only=True)
                logger.info("成功从本地路径加载模型")
            except Exception as e:
                logger.warning(f"从本地路径加载模型失败: {str(e)}")
        
        # 尝试从预训练模型名称加载
        if model is None:
            logger.info(f"从预训练模型加载模型: {model_name}")
            try:
                # 先尝试本地加载
                model = QueryClassifierModel(model_name, num_labels=2, local_files_only=True)
                logger.info(f"成功从本地加载预训练模型: {model_name}")
            except Exception as e:
                logger.warning(f"从本地加载预训练模型失败，尝试在线加载: {str(e)}")
                try:
                    # 再尝试在线加载
                    model = QueryClassifierModel(model_name, num_labels=2, local_files_only=False)
                    logger.info(f"成功从在线加载预训练模型: {model_name}")
                except Exception as e2:
                    logger.warning(f"从在线加载预训练模型失败: {str(e2)}")
        
        # 如果所有尝试都失败，创建一个新的模型
        if model is None:
            logger.warning("所有预训练模型加载失败，使用随机初始化的模型")
            model = QueryClassifierModel("bert-base-uncased", num_labels=2, local_files_only=False)
    
    except Exception as e:
        logger.error(f"所有模型加载方法都失败: {str(e)}")
        raise RuntimeError("无法加载模型")
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",  # 如果有验证集，可以设置为 "epoch"
        load_best_model_at_end=False,
        save_total_limit=2,
    )
    
    # 定义数据整理函数
    def data_collator(features):
        batch = {}
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.stack([f["token_type_ids"] for f in features])
        if "labels" in features[0]:
            batch["labels"] = torch.stack([f["labels"] for f in features])
        return batch
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始训练模型...")
    trainer.train()
    
    # 保存模型和分词器
    logger.info(f"保存模型到 {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 保存标签映射
    with open(os.path.join(output_dir, "label_maps.json"), "w", encoding="utf-8") as f:
        json.dump({"label_maps": LABEL_MAPS, "reverse_label_maps": REVERSE_LABEL_MAPS}, f, ensure_ascii=False, indent=2)
    
    logger.info("模型训练完成")
    return model, tokenizer

class QueryClassifier:
    """
    查询分类器类，用于加载训练好的模型并进行推理
    """
    def __init__(self, model_path: str = "./query_classifier_model"):
        """
        初始化分类器
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 检查环境变量中是否指定了模型路径
        env_model_path = os.environ.get("BERT_MODEL_PATH", "")
        
        # 加载模型和分词器
        logger.info(f"从 {model_path} 加载模型")
        
        # 初始化模型和分词器
        self.model = None
        self.tokenizer = None
        
        # 尝试加载模型的顺序：
        # 1. 从指定路径加载预训练模型
        # 2. 从环境变量指定的路径加载
        # 3. 使用QueryClassifierModel直接初始化
        # 4. 使用简单分词器和随机初始化模型
        
        # 1. 尝试从指定路径加载预训练模型
        try:
            logger.info(f"尝试从 {model_path} 加载预训练模型")
            # 尝试加载模型
            try:
                self.model = QueryClassifierModel.from_pretrained(model_path)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"成功从 {model_path} 加载预训练模型")
            except Exception as e:
                logger.warning(f"从 {model_path} 加载预训练模型失败: {str(e)}")
            
            # 尝试加载分词器
            if self.model is not None:
                try:
                    self.tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
                    logger.info(f"成功从 {model_path} 加载分词器")
                except Exception as e:
                    logger.warning(f"从 {model_path} 加载分词器失败: {str(e)}")
                    self.tokenizer = None
        except Exception as e:
            logger.warning(f"从指定路径加载预训练模型时出错: {str(e)}")
        
        # 2. 尝试从环境变量指定的路径加载
        if self.model is None or self.tokenizer is None:
            if env_model_path and os.path.exists(env_model_path):
                logger.info(f"尝试从环境变量指定的路径加载: {env_model_path}")
                try:
                    if self.model is None:
                        try:
                            self.model = QueryClassifierModel.from_pretrained(env_model_path)
                            self.model.to(self.device)
                            self.model.eval()
                            logger.info(f"成功从环境变量指定的路径加载模型")
                        except Exception as e:
                            logger.warning(f"从环境变量指定的路径加载模型失败: {str(e)}")
                    
                    if self.tokenizer is None:
                        try:
                            self.tokenizer = BertTokenizer.from_pretrained(env_model_path, local_files_only=True)
                            logger.info(f"成功从环境变量指定的路径加载分词器")
                        except Exception as e:
                            logger.warning(f"从环境变量指定的路径加载分词器失败: {str(e)}")
                except Exception as e:
                    logger.warning(f"从环境变量指定的路径加载时出错: {str(e)}")
        
        # 3. 使用QueryClassifierModel直接初始化
        if self.model is None:
            logger.info("尝试使用QueryClassifierModel直接初始化模型")
            try:
                # 先尝试本地加载
                self.model = QueryClassifierModel(model_path, num_labels=2, local_files_only=True)
                self.model.to(self.device)
                self.model.eval()
                logger.info("成功使用QueryClassifierModel初始化模型")
            except Exception as e:
                logger.warning(f"使用QueryClassifierModel初始化模型失败: {str(e)}")
                try:
                    # 尝试使用默认配置初始化
                    self.model = QueryClassifierModel("bert-base-uncased", num_labels=2, local_files_only=False)
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info("成功使用默认配置初始化模型")
                except Exception as e2:
                    logger.error(f"使用默认配置初始化模型也失败: {str(e2)}")
                    raise RuntimeError(f"无法初始化模型: {str(e)}")
        
        # 尝试加载分词器，如果失败则使用备用分词器或简单分词器
        if self.tokenizer is None:
            logger.info("尝试加载备用分词器")
            backup_models = ["bert-base-chinese", "bert-base-multilingual-uncased", "distilbert-base-uncased"]
            for backup_model in backup_models:
                try:
                    # 先尝试本地加载
                    self.tokenizer = BertTokenizer.from_pretrained(backup_model, local_files_only=True)
                    logger.info(f"成功从本地加载备用分词器: {backup_model}")
                    break
                except Exception:
                    try:
                        # 再尝试在线加载
                        self.tokenizer = BertTokenizer.from_pretrained(backup_model, local_files_only=False)
                        logger.info(f"成功从在线加载备用分词器: {backup_model}")
                        break
                    except Exception as e:
                        logger.warning(f"加载备用分词器 {backup_model} 失败: {str(e)}")
            
            # 如果所有备用分词器都加载失败，使用简单分词器
            if self.tokenizer is None:
                from transformers import BasicTokenizer
                logger.warning("所有备用分词器加载失败，使用简单分词器替代")
                
                # 创建一个简单的分词器包装类，模拟BertTokenizer的接口
                class SimpleTokenizer:
                    def __init__(self):
                        self.basic_tokenizer = BasicTokenizer(do_lower_case=True)
                        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
                        self.max_len = 512
                    
                    def __call__(self, text, return_tensors=None, truncation=True, padding='max_length', max_length=128):
                        # 简单分词
                        tokens = ["[CLS]"] + self.basic_tokenizer.tokenize(text)[:max_length-2] + ["[SEP]"]
                        
                        # 转换为ID
                        input_ids = []
                        for token in tokens:
                            if token in self.vocab:
                                input_ids.append(self.vocab[token])
                            else:
                                # 为新词分配ID
                                self.vocab[token] = len(self.vocab)
                                input_ids.append(self.vocab[token])
                        
                        # 填充
                        if padding == 'max_length':
                            attention_mask = [1] * len(input_ids)
                            padding_length = max_length - len(input_ids)
                            input_ids = input_ids + [0] * padding_length
                            attention_mask = attention_mask + [0] * padding_length
                        
                        # 转换为张量
                        if return_tensors == 'pt':
                            import torch
                            return {
                                "input_ids": torch.tensor([input_ids]),
                                "attention_mask": torch.tensor([attention_mask])
                            }
                        
                        return {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask
                        }
                    
                    def save_pretrained(self, path):
                        os.makedirs(path, exist_ok=True)
                        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
                            json.dump({"type": "simple_tokenizer"}, f)
                
                self.tokenizer = SimpleTokenizer()
                logger.info("已初始化简单分词器")
        
        # 加载标签映射
        label_maps_path = os.path.join(model_path, "label_maps.json")
        if os.path.exists(label_maps_path):
            with open(label_maps_path, "r", encoding="utf-8") as f:
                label_maps_data = json.load(f)
                self.reverse_label_maps = label_maps_data.get("reverse_label_maps", REVERSE_LABEL_MAPS)
        else:
            logger.warning(f"标签映射文件 {label_maps_path} 不存在，使用默认映射")
            self.reverse_label_maps = REVERSE_LABEL_MAPS
        
        logger.info("查询分类器初始化完成")
    
    def predict(self, query: str) -> Dict[str, str]:
        """
        对查询进行分类预测
        
        Args:
            query: 用户查询文本
            
        Returns:
            包含预测标签的字典 {"query_strategy": "noRAG"/"hybrid", "precision_required": "yes"/"no"}
        """
        # 对查询进行编码
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 进行推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 获取logits并应用sigmoid
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)
        
        # 使用阈值0.5进行预测
        predictions = (probs > 0.5).int().cpu().numpy()[0]
        
        # 将预测结果转换为标签文本
        result = {
            "query_strategy": self.reverse_label_maps["query_strategy"][predictions[0]],
            "precision_required": self.reverse_label_maps["precision_required"][predictions[1]]
        }
        
        return result
        
    def predict_with_confidence(self, query: str) -> Tuple[str, float]:
        """
        对查询进行分类预测并返回置信度
        
        Args:
            query: 用户查询文本
            
        Returns:
            (label, confidence): 预测标签和置信度的元组
        """
        # 对查询进行编码
        try:
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 进行推理
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 获取logits并应用sigmoid
            logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            # 使用阈值0.5进行预测
            predictions = (probs > 0.5).astype(int)
            
            # 获取置信度（使用概率值）
            confidences = probs.copy()
            # 对于小于0.5的概率，使用1-p作为置信度
            confidences[confidences < 0.5] = 1 - confidences[confidences < 0.5]
            
            # 计算平均置信度
            avg_confidence = float(np.mean(confidences))
            
            # 将预测结果转换为标签文本
            query_strategy = self.reverse_label_maps["query_strategy"][predictions[0]]
            precision_required = self.reverse_label_maps["precision_required"][predictions[1]]
            
            # 组合标签
            label = f"{query_strategy}:{precision_required}"
            
            return label, avg_confidence
        except Exception as e:
            logger.error(f"预测查询时出错: {str(e)}")
            return "noRAG:no", 0.5  # 返回默认标签和中等置信度

def main():
    """
    主函数，用于测试模型训练和推理
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="查询分类器训练和推理")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--predict", type=str, help="对查询进行预测")
    parser.add_argument("--data_path", type=str, default="./data/query_classifier_data.jsonl", help="训练数据路径")
    parser.add_argument("--model_path", type=str, default="./query_classifier_model", help="模型保存/加载路径")
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args.data_path, args.model_path)
    
    if args.predict:
        classifier = QueryClassifier(args.model_path)
        result = classifier.predict(args.predict)
        print(f"查询: {args.predict}")
        print(f"预测结果: {result}")

if __name__ == "__main__":
    main()
