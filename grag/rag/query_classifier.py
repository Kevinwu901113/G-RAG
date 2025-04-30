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
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        初始化模型
        
        Args:
            model_name: 预训练模型名称
            num_labels: 标签数量
        """
        super(QueryClassifierModel, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
    
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

def train_model(data_path: str, output_dir: str = "./query_classifier_model", batch_size: int = 8, num_epochs: int = 5):
    """
    训练多标签分类模型
    
    Args:
        data_path: 训练数据路径
        output_dir: 模型输出目录
        batch_size: 批次大小
        num_epochs: 训练轮数
    """
    # 加载数据
    texts, labels = load_data(data_path)
    
    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 创建数据集
    dataset = QueryDataset(texts, labels, tokenizer)
    
    # 初始化模型
    model = QueryClassifierModel("bert-base-uncased", num_labels=2)
    
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
        
        # 加载模型和分词器
        logger.info(f"从 {model_path} 加载模型")
        self.model = QueryClassifierModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 加载标签映射
        label_maps_path = os.path.join(model_path, "label_maps.json")
        if os.path.exists(label_maps_path):
            with open(label_maps_path, "r", encoding="utf-8") as f:
                label_maps_data = json.load(f)
                self.reverse_label_maps = label_maps_data.get("reverse_label_maps", REVERSE_LABEL_MAPS)
        else:
            logger.warning(f"标签映射文件 {label_maps_path} 不存在，使用默认映射")
            self.reverse_label_maps = REVERSE_LABEL_MAPS
    
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