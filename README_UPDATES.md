# GRAG项目更新说明

## 新增功能

### 1. 实体提取和关系提取增强

基于LightRAG项目的实现，我们对GRAG项目的实体提取和关系提取功能进行了增强：

- 新增了专门的实体提取模块 `grag/rag/entity_extraction.py`，提供了更强大的实体和关系提取能力
- 支持多轮提取（gleaning），可以从文本中提取更多的实体和关系
- 改进了实体和关系的合并逻辑，支持使用LLM进行描述合并
- 增加了文件路径追踪，便于溯源

### 2. 日志管理系统

新增了专门的日志管理模块 `grag/utils/logger_manager.py`，提供了更完善的日志功能：

- 单例模式的日志管理器，统一管理项目中的所有日志
- 支持按会话创建日志文件，便于追踪和调试
- 提供系统信息、配置信息、操作信息、性能信息和错误信息的记录
- 支持动态调整日志级别
- 支持多日志器管理，可以为不同模块创建不同的日志器

## 使用方法

### 实体提取和关系提取

实体提取和关系提取功能已经集成到 `LightRAG` 类中，无需额外调用。当调用 `index_document` 方法时，会自动进行实体和关系提取。

如果需要单独使用实体提取功能，可以直接调用 `entity_extraction` 模块中的函数：

```python
from grag.rag.entity_extraction import extract_entities_and_relationships, process_and_store_entities_relationships

# 提取实体和关系
entities, relationships = await extract_entities_and_relationships(
    content="文本内容",
    chunk_key="文档ID",
    llm_model_func=llm_model_func,
    llm_model_kwargs=llm_model_kwargs,
    knowledge_graph_inst=graph_storage,
    entity_vdb=entity_vector_storage,
    relationships_vdb=relationship_vector_storage,
    file_path="文件路径",
    max_gleaning=1  # 额外提取次数
)

# 处理并存储实体和关系
await process_and_store_entities_relationships(
    entities=entities,
    relationships=relationships,
    knowledge_graph_inst=graph_storage,
    entity_vdb=entity_vector_storage,
    relationships_vdb=relationship_vector_storage,
    force_llm_summary_on_merge=6,  # 合并阈值
    llm_model_func=llm_model_func,
    llm_model_kwargs=llm_model_kwargs
)
```

### 日志管理

日志管理系统已经集成到项目中，可以通过以下方式使用：

```python
from grag.utils.logger_manager import get_logger, configure_logging

# 配置日志系统
log_manager = configure_logging(log_dir="logs", default_level="info")

# 记录系统信息
log_manager.log_system_info()

# 记录配置信息
log_manager.log_config(config_dict)

# 获取日志器
logger = get_logger("模块名称")

# 记录日志
logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")

# 记录操作信息
log_manager.log_operation("操作名称", {"参数1": "值1", "参数2": "值2"})

# 记录性能信息
start_time = time.time()
# 执行操作
end_time = time.time()
log_manager.log_performance("操作名称", start_time, end_time)

# 记录错误信息
try:
    # 执行操作
except Exception as e:
    log_manager.log_error(e, "错误上下文")
```

## 配置说明

### 实体提取配置

在 `config.yaml` 中可以添加以下配置项：

```yaml
entity_extraction:
  max_gleaning: 1  # 额外提取次数
  force_llm_summary_on_merge: 6  # 合并阈值
  entity_types:  # 实体类型列表
    - 组织名称
    - 个人姓名
    - 地理位置
    - 事件
    - 时间
    - 职位
    - 金额
    - 面积
    - 人数
```

### 日志配置

在 `config.yaml` 中可以添加以下配置项：

```yaml
logging:
  log_dir: logs  # 日志目录
  default_level: info  # 默认日志级别
  file_prefix: grag  # 日志文件前缀
```

## 注意事项

1. 实体提取和关系提取功能依赖于LLM模型，请确保配置了正确的LLM模型。
2. 日志文件会随着时间增长，请定期清理日志目录。
3. 实体提取和关系提取可能会消耗较多的计算资源，请根据实际情况调整配置。
