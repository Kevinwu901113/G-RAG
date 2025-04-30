# 查询分类器模块

## 概述

查询分类器是一个多标签文本分类器，用于根据用户输入的查询判断两件事：

1. 该查询应该走哪种检索路径：`noRAG`还是`hybrid`
2. 该查询是否需要精度支持：`yes`还是`no`

最终模型的输出是两个标签：
- `query_strategy`（取值为`noRAG`或`hybrid`）
- `precision_required`（取值为`yes`或`no`）

## 模块结构

查询分类器模块包含以下主要组件：

1. **数据加载模块**：处理JSONL格式的训练数据，将文本标签转换为数字标签
2. **模型定义**：基于BERT的多标签分类模型
3. **训练脚本**：用于训练模型的脚本
4. **推理脚本**：用于加载训练好的模型并进行预测的脚本
5. **示例数据生成脚本**：用于生成测试数据的辅助脚本

## 技术实现

- 使用`bert-base-uncased`作为编码器
- 使用`BCEWithLogitsLoss`作为损失函数，适用于多标签分类任务
- 使用`sigmoid`函数和阈值（默认0.5）判断每个标签
- 模型保存为Huggingface标准格式，包含模型参数和tokenizer

## 标签映射

训练数据中的文本标签会被转换为数字标签：

```
query_strategy: noRAG -> 0, hybrid -> 1
precision_required: no -> 0, yes -> 1
```

## 使用方法

### 1. 生成示例数据（可选）

如果没有训练数据，可以使用示例数据生成脚本生成测试数据：

```bash
python scripts/generate_sample_data.py --output_path ./data/query_classifier_sample_data.jsonl --num_samples 100
```

### 2. 训练模型

使用训练脚本训练模型：

```bash
python scripts/train_query_classifier.py --data_path ./data/query_classifier_sample_data.jsonl --output_dir ./query_classifier_model --batch_size 8 --epochs 5
```

查看数据样例（不训练模型）：

```bash
python scripts/train_query_classifier.py --data_path ./data/query_classifier_sample_data.jsonl --sample_data
```

### 3. 使用模型进行预测

使用推理脚本对查询进行预测：

```bash
python scripts/predict_query.py --query "请分析这篇文章的主要观点" --model_path ./query_classifier_model
```

### 4. 在代码中使用

在Python代码中使用查询分类器：

```python
from grag.rag.query_classifier import QueryClassifier

# 初始化分类器
classifier = QueryClassifier(model_path="./query_classifier_model")

# 进行预测
result = classifier.predict("请分析这篇文章的主要观点")
print(result)  # {'query_strategy': 'hybrid', 'precision_required': 'no'}
```

## 训练数据格式

训练数据使用JSONL格式，每行包含一个JSON对象，包含以下字段：

```json
{
  "query": "用户查询文本",
  "query_strategy": "noRAG或hybrid",
  "precision_required": "yes或no"
}
```

## 注意事项

1. 确保训练数据的标签处理和推理时的一致性
2. 模型使用多标签分类任务的损失函数（BCEWithLogitsLoss）
3. 推理时使用sigmoid+阈值判断每个标签，而不是使用argmax
4. 代码设计支持未来插入增量样本再训练，没有硬编码的样本结构假设

## 文件说明

- `grag/rag/query_classifier.py`: 核心模块，包含数据加载、模型定义和训练函数
- `scripts/train_query_classifier.py`: 训练脚本
- `scripts/predict_query.py`: 推理脚本
- `scripts/generate_sample_data.py`: 示例数据生成脚本