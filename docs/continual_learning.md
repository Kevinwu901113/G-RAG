# 持续学习模块使用文档

## 简介

持续学习模块是G-RAG项目的一个可插拔组件，用于定期更新已实现的多标签查询分类器。该模块基于用户交互日志或人工标注数据进行增量训练，支持无中断地更新分类模型。

## 模块结构

- `grag/classifier/continual_trainer.py`: 核心持续学习模块，实现从指定路径加载新数据并对查询分类器进行微调
- `scripts/run_continual_update.py`: 调度脚本，用于周期性地运行持续学习模块

## 配置参数

持续学习模块支持通过配置文件或命令行参数进行配置。以下是可配置的参数：

### 在config.yaml中配置

```yaml
# 持续学习模块配置示例
continual_learning:
  # 新数据缓冲区路径
  buffer_path: "logs/continual_buffer.jsonl"
  # 模型路径
  model_path: "./query_classifier_model"
  # 模型备份目录
  backup_model_dir: "./model_backups"
  # 训练批次大小
  batch_size: 8
  # 训练轮数
  epochs: 3
  # 是否保存旧模型备份
  save_backup: true
  # 更新间隔（秒）
  update_interval_seconds: 3600
```

## 数据格式

持续学习模块要求新数据与原训练数据格式完全一致，必须包含以下字段：

- `query`: 用户查询文本
- `query_strategy`: 查询策略，可选值为 "noRAG" 或 "hybrid"
- `precision_required`: 精度要求，可选值为 "no" 或 "yes"

示例数据格式：

```json
{"query": "什么是G-RAG？", "query_strategy": "hybrid", "precision_required": "yes"}
{"query": "今天天气怎么样？", "query_strategy": "noRAG", "precision_required": "no"}
```

## 使用方法

### 作为独立脚本运行

```bash
# 运行一次持续学习更新
python scripts/run_continual_update.py --run_once

# 以指定间隔运行持续学习更新
python scripts/run_continual_update.py --interval 7200  # 每2小时运行一次

# 指定配置文件路径
python scripts/run_continual_update.py --config custom_config.yaml
```

### 在代码中集成调用

```python
from grag.classifier.continual_trainer import run_continual_learning

# 使用默认配置运行持续学习
success = run_continual_learning()

# 使用自定义配置运行持续学习
success = run_continual_learning(
    config_path="config.yaml",
    buffer_path="custom_buffer.jsonl",
    batch_size=16,
    epochs=5
)
```

## 错误处理

持续学习模块会在以下情况抛出异常：

- 数据格式错误：当新数据不符合预期格式时
- 路径不存在：当模型路径或数据缓冲区路径不存在时
- 标签缺失：当数据中缺少必要的标签字段时

所有异常都会被记录到日志中，并且不会导致主流程中断。

## 注意事项

1. 持续学习模块不会改变已有模型的标签顺序、向量维度或输出结构，只进行继续训练。
2. 模型保存使用Huggingface标准格式，与原有系统结构兼容。
3. 建议定期清理模型备份目录，避免占用过多磁盘空间。