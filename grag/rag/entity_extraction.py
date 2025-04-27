import asyncio
import re
import json
from typing import Dict, List, Any, Tuple, Optional, Set, DefaultDict
from collections import defaultdict

from ..core.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, TextChunkSchema
from ..core.prompt import PROMPTS, GRAPH_FIELD_SEP
from ..utils.common import (
    compute_mdhash_id,
    logger,
    clean_str,
    split_string_by_multi_markers,
    create_progress_bar,
)

async def extract_entities_and_relationships(
    content: str,
    chunk_key: str,
    llm_model_func: callable,
    llm_model_kwargs: Dict[str, Any],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    file_path: str = "unknown_source",
    max_gleaning: int = 2,
    timeout: int = 180,  # 增加默认超时时间参数，默认为180秒，与LLM客户端超时一致
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从文本内容中提取实体和关系
    
    Args:
        content: 文本内容
        chunk_key: 文本块ID
        llm_model_func: LLM模型函数
        llm_model_kwargs: LLM模型参数
        knowledge_graph_inst: 知识图谱存储
        entity_vdb: 实体向量存储
        relationships_vdb: 关系向量存储
        file_path: 文件路径
        max_gleaning: 最大提取次数
        timeout: LLM调用的超时时间（秒）
        
    Returns:
        提取的实体和关系列表
    """
    logger.info(f"开始从文本块 {chunk_key} 提取实体和关系")
    
    # 1. 准备提示模板和上下文基础信息
    context_base = {
        "entity_types": ", ".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        "input_text": content,
        "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    }
    
    entity_extraction_prompt = PROMPTS["entity_extraction"].format(**context_base)
    
    # 确保llm_model_kwargs中包含timeout参数
    model_kwargs = llm_model_kwargs.copy()
    model_kwargs["timeout"] = model_kwargs.get("timeout", timeout)
    
    # 2. 提取实体和关系，使用asyncio.wait_for添加额外的超时保护
    retry_count = 0
    max_retries = 3
    retry_delay = 2
    
    while retry_count < max_retries:
        try:
            final_result = await asyncio.wait_for(
                llm_model_func(entity_extraction_prompt, **model_kwargs),
                timeout=timeout
            )
            
            # 存储消息历史以便后续提取
            history = [
                {"role": "user", "content": entity_extraction_prompt},
                {"role": "assistant", "content": final_result}
            ]
            break  # 成功获取结果，跳出重试循环
        except asyncio.TimeoutError:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"从文本块 {chunk_key} 提取实体和关系超时（超过 {timeout} 秒），已重试 {max_retries} 次")
                return [], []  # 超时且重试失败时返回空列表
            
            logger.warning(f"从文本块 {chunk_key} 提取实体和关系超时，正在进行第 {retry_count} 次重试...")
            await asyncio.sleep(retry_delay * retry_count)  # 指数退避策略
        except Exception as e:
            logger.error(f"从文本块 {chunk_key} 提取实体和关系出错: {str(e)}")
            return [], []  # 其他错误时返回空列表
    
    # 3. 实体提取循环（gleaning）- 增强多轮提取机制
    entity_extraction_complete = False
    for now_glean_index in range(max_gleaning):
        # 继续提取实体
        continue_prompt = PROMPTS["entiti_continue_extraction"]
        try:
            # 使用相同的超时保护机制
            glean_result = await asyncio.wait_for(
                llm_model_func(continue_prompt, history_messages=history, **model_kwargs),
                timeout=timeout
            )
            
            # 检查结果是否为空或无意义
            if not glean_result.strip() or glean_result.strip() == PROMPTS["DEFAULT_COMPLETION_DELIMITER"]:
                entity_extraction_complete = True
                break
                
            # 更新历史和结果
            history.append({"role": "user", "content": continue_prompt})
        except asyncio.TimeoutError:
            logger.error(f"从文本块 {chunk_key} 提取实体和关系的gleaning过程超时（超过 {timeout} 秒）")
            break  # 超时时中断gleaning循环
        history.append({"role": "assistant", "content": glean_result})
        final_result += glean_result
        
        # 如果已达到最大提取次数，停止循环
        if now_glean_index == max_gleaning - 1:
            break

        # 检查是否需要继续提取实体
        if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
        try:
            if_loop_result = await asyncio.wait_for(
                llm_model_func(if_loop_prompt, history_messages=history, **llm_model_kwargs),
                timeout=timeout
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                entity_extraction_complete = True
                break
        except (asyncio.TimeoutError, Exception) as e:
            logger.error(f"检查是否继续提取实体时出错: {str(e)}")
            entity_extraction_complete = True
            break  # 出错时停止提取
    
    # 4. 关系提取循环 - 独立进行关系提取以获取更多关系
    relation_extraction_complete = False
    for now_glean_index in range(max_gleaning + 1):  # 增加关系提取的轮次
        # 继续提取关系
        re_continue_prompt = PROMPTS["relation_continue_extraction"]
        
        # 添加重试逻辑，与实体提取部分保持一致
        retry_count = 0
        max_retries = 3
        retry_delay = 2
        
        while retry_count < max_retries:
            try:
                glean_result = await asyncio.wait_for(
                    llm_model_func(re_continue_prompt, history_messages=history, **llm_model_kwargs),
                    timeout=timeout
                )
                break  # 成功获取结果，跳出重试循环
            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"从文本块 {chunk_key} 提取关系超时（超过 {timeout} 秒），已重试 {max_retries} 次")
                    relation_extraction_complete = True
                    break  # 超时且重试失败时中断关系提取
                
                logger.warning(f"从文本块 {chunk_key} 提取关系超时，正在进行第 {retry_count} 次重试...")
                await asyncio.sleep(retry_delay * retry_count)  # 指数退避策略
            except Exception as e:
                logger.error(f"从文本块 {chunk_key} 提取关系出错: {str(e)}")
                relation_extraction_complete = True
                break  # 其他错误时中断关系提取
        
        # 如果重试失败，中断关系提取循环
        if retry_count >= max_retries:
            break
            
        # 检查结果是否为空或无意义
        if not glean_result.strip() or glean_result.strip() == PROMPTS["DEFAULT_COMPLETION_DELIMITER"]:
            relation_extraction_complete = True
            break
            
        # 更新历史和结果
        history.append({"role": "user", "content": re_continue_prompt})
        history.append({"role": "assistant", "content": glean_result})
        final_result += glean_result
        
        # 如果已达到最大提取次数，停止循环
        if now_glean_index == max_gleaning:
            break

        # 检查是否需要继续提取关系
        re_if_loop_prompt = PROMPTS["relation_if_loop_extraction"]
        try:
            if_loop_result = await asyncio.wait_for(
                llm_model_func(re_if_loop_prompt, history_messages=history, **llm_model_kwargs),
                timeout=timeout
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                relation_extraction_complete = True
                break
        except (asyncio.TimeoutError, Exception) as e:
            logger.error(f"检查是否继续提取关系时出错: {str(e)}")
            relation_extraction_complete = True
            break  # 出错时停止提取
    
    # 5. 解析最终结果
    entities, relationships = await _parse_extraction_response(final_result, chunk_key, file_path)
    
    # 6. 记录提取状态
    extraction_status = []
    if entity_extraction_complete:
        extraction_status.append("实体提取完成")
    else:
        extraction_status.append("实体提取未完全完成（达到最大轮次限制）")
        
    if relation_extraction_complete:
        extraction_status.append("关系提取完成")
    else:
        extraction_status.append("关系提取未完全完成（达到最大轮次限制）")
    
    logger.info(f"从文本块 {chunk_key} 提取完成，共 {len(entities)} 个实体和 {len(relationships)} 个关系。状态: {', '.join(extraction_status)}")
    return entities, relationships


async def process_chunks_parallel(
    chunks: Dict[str, TextChunkSchema],
    llm_model_func: callable,
    llm_model_kwargs: Dict[str, Any],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    max_gleaning: int = 3,
    timeout_per_chunk: int = 360,  # 每个块的处理超时时间（秒），增加以适应更长的LLM处理时间
    max_concurrent_tasks: int = 5,  # 最大并发任务数
) -> BaseGraphStorage:
    """
    并行处理多个文本块，提取实体和关系，支持超时控制和进度监控
    
    Args:
        chunks: 文本块字典，键为块ID，值为块内容
        llm_model_func: LLM模型函数
        llm_model_kwargs: LLM模型参数
        knowledge_graph_inst: 知识图谱存储
        entity_vdb: 实体向量存储
        relationships_vdb: 关系向量存储
        max_gleaning: 最大提取次数
        timeout_per_chunk: 每个块的处理超时时间（秒）
        max_concurrent_tasks: 最大并发任务数，控制内存使用
        
    Returns:
        更新后的知识图谱存储实例
    """
    logger.info(f"开始并行处理 {len(chunks)} 个文本块进行实体和关系提取，最大并发数: {max_concurrent_tasks}")
    
    # 初始化进度计数器和进度条
    total_chunks = len(chunks)
    already_processed = 0
    already_entities = 0
    already_relations = 0
    failed_chunks = 0
    
    # 创建进度条
    progress_bar = create_progress_bar(total=total_chunks, desc="实体关系提取")
    
    # 按块ID排序处理块
    ordered_chunks = [(k, v) for k, v in chunks.items()]
    
    # 创建信号量控制并发任务数量
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    # 定义单个内容处理函数
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations, failed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        file_path = chunk_dp.get("file_path", "unknown_source")
        
        # 使用信号量控制并发
        async with semaphore:
            try:
                # 直接传递timeout参数给extract_entities_and_relationships函数
                # 不再需要额外的asyncio.wait_for，因为函数内部已经实现了超时处理
                entities, relationships = await extract_entities_and_relationships(
                    content,
                    chunk_key,
                    llm_model_func,
                    llm_model_kwargs,
                    knowledge_graph_inst,
                    entity_vdb,
                    relationships_vdb,
                    file_path,
                    max_gleaning,
                    timeout=timeout_per_chunk
                )
                
                # 更新计数器并显示进度
                already_processed += 1
                already_entities += len(entities)
                already_relations += len(relationships)
                
                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "已处理": already_processed,
                    "实体": already_entities,
                    "关系": already_relations,
                    "失败": failed_chunks
                })
                
                return entities, relationships
            except asyncio.TimeoutError:
                # 处理超时情况
                logger.warning(f"处理文本块 {chunk_key} 超时（超过 {timeout_per_chunk} 秒）")
                failed_chunks += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "已处理": already_processed,
                    "失败": failed_chunks
                })
                return [], []
            except Exception as e:
                # 处理其他异常
                logger.error(f"处理文本块 {chunk_key} 出错: {str(e)}")
                failed_chunks += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "已处理": already_processed,
                    "失败": failed_chunks
                })
                return [], []
    # 创建并执行所有任务
    tasks = [_process_single_content(chunk) for chunk in ordered_chunks]
    results = await asyncio.gather(*tasks)
    
    # 处理结果
    all_entities = []
    all_relationships = []
    for entities, relationships in results:
        all_entities.extend(entities)
        all_relationships.extend(relationships)
    
    # 关闭进度条
    progress_bar.close()
    
    # 输出处理结果统计
    logger.info(f"实体关系提取完成，共处理 {already_processed} 个文本块，提取 {already_entities} 个实体和 {already_relations} 个关系")
    if failed_chunks > 0:
        logger.warning(f"处理过程中有 {failed_chunks} 个文本块失败")
        
    # 处理并存储提取的实体和关系
    if all_entities or all_relationships:
        await process_and_store_entities_relationships(
            entities=all_entities,
            relationships=all_relationships,
            knowledge_graph_inst=knowledge_graph_inst,
            entity_vdb=entity_vdb,
            relationships_vdb=relationships_vdb,
            force_llm_summary_on_merge=6,  # 默认合并阈值
            llm_model_func=llm_model_func,
            llm_model_kwargs=llm_model_kwargs,
        )
        logger.info("实体和关系处理与存储完成")
    
    # 返回更新后的知识图谱
    return knowledge_graph_inst
    
    # 处理结果
    all_entities = []
    all_relationships = []
    for entities, relationships in results:
        all_entities.extend(entities)
        all_relationships.extend(relationships)
    
    # 关闭进度条
    progress_bar.close()
    
    # 输出处理结果统计
    logger.info(f"实体关系提取完成，共处理 {already_processed} 个文本块，提取 {already_entities} 个实体和 {already_relations} 个关系")
    if failed_chunks > 0:
        logger.warning(f"处理过程中有 {failed_chunks} 个文本块失败")
        
    # 处理并存储提取的实体和关系
    if all_entities or all_relationships:
        await process_and_store_entities_relationships(
            entities=all_entities,
            relationships=all_relationships,
            knowledge_graph_inst=knowledge_graph_inst,
            entity_vdb=entity_vdb,
            relationships_vdb=relationships_vdb,
            force_llm_summary_on_merge=6,  # 默认合并阈值
            llm_model_func=llm_model_func,
            llm_model_kwargs=llm_model_kwargs,
        )
        logger.info("实体和关系处理与存储完成")
    
    # 返回更新后的知识图谱
    return knowledge_graph_inst

async def _parse_extraction_response(
    response: str, chunk_key: str, file_path: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    解析LLM返回的实体和关系提取结果
    
    Args:
        response: LLM返回的响应
        chunk_key: 文本块ID
        file_path: 文件路径
        
    Returns:
        提取的实体和关系列表
    """
    entities = []
    relationships = []
    
    # 按记录分隔符分割
    records = response.split(PROMPTS["DEFAULT_RECORD_DELIMITER"])
    
    for record in records:
        record = record.strip()
        if not record:
            continue
            
        # 解析记录
        parts = record.split(PROMPTS["DEFAULT_TUPLE_DELIMITER"])
        if len(parts) < 2:
            continue
            
        record_type = parts[0].strip('(")')
        
        # 处理实体记录
        if record_type == "entity":
            entity = await _handle_entity_record(parts, chunk_key, file_path)
            if entity:
                entities.append(entity)
                
        # 处理关系记录
        elif record_type == "relationship":
            relationship = await _handle_relationship_record(parts, chunk_key, file_path)
            if relationship:
                relationships.append(relationship)
    
    return entities, relationships

async def _handle_entity_record(
    parts: List[str], chunk_key: str, file_path: str
) -> Optional[Dict[str, Any]]:
    """处理实体记录"""
    if len(parts) < 5:
        return None
        
    try:
        node_type = parts[1].strip()
        node_name = parts[2].strip()
        name_type = parts[3].strip()
        attributes_str = parts[4].strip().rstrip(')')
        
        # 确保实体类型不是UNKNOWN
        if name_type == "UNKNOWN" or not name_type:
            # 根据实体名称和属性推断类型
            name_type = _infer_entity_type(node_name, attributes_str)
        
        # 尝试解析属性JSON
        try:
            # 处理可能的JSON格式问题
            attributes_str = attributes_str.replace("'", '"')
            attributes = json.loads(attributes_str)
        except json.JSONDecodeError:
            # 如果JSON解析失败，使用简单的描述
            attributes = {"other": attributes_str}
        
        # 构建实体描述
        description = ""
        if isinstance(attributes, dict):
            if "time" in attributes:
                description += f"时间: {attributes['time']}; "
            if "place" in attributes:
                description += f"地点: {attributes['place']}; "
            if "event" in attributes:
                description += f"事件: {attributes['event']}; "
            if "other" in attributes:
                description += f"其他: {attributes['other']}; "
        else:
            description = str(attributes)
        
        # 如果描述为空，添加默认描述
        if not description.strip():
            description = f"{name_type}类型的实体"
            
        # 生成唯一ID
        entity_id = compute_mdhash_id(node_name, prefix="ent-")
            
        return {
            "id": entity_id,
            "name": node_name,
            "type": name_type,
            "description": description.strip("; "),
            "entity_type": name_type,  # 保留原有字段以兼容现有代码
            "source_id": chunk_key,
            "file_path": file_path,
        }
    except Exception as e:
        logger.warning(f"处理实体记录时出错: {e}, 记录: {parts}")
        return None

def _infer_entity_type(node_name: str, attributes_str: str) -> str:
    """
    根据实体名称和属性推断实体类型
    
    Args:
        node_name: 实体名称
        attributes_str: 属性字符串
        
    Returns:
        推断的实体类型
    """
    # 默认实体类型
    entity_types = PROMPTS["DEFAULT_ENTITY_TYPES"]
    
    # 检查是否包含年份或日期格式
    if re.search(r'\d{4}年|\d{2}年|y\d{4}|^\d{4}$|\d+月|\d+日|季度', node_name):
        return "时间"
    
    # 检查是否是人名（通常是2-4个字的中文名字）
    if re.match(r'^[\u4e00-\u9fa5]{2,4}$', node_name) and not re.search(r'镇|县|市|省|区|村|街道|路|公司|组织|部门', node_name):
        return "个人姓名"
    
    # 检查是否是地理位置
    if re.search(r'镇|县|市|省|区|村|街道|路|地区|国家|大厦|广场|中心|园区', node_name):
        return "地理位置"
    
    # 检查是否是组织名称
    if re.search(r'公司|集团|组织|部门|委员会|学校|大学|医院|机构|协会|联盟|局|厅|部|处|司|院|所|中心|办公室', node_name):
        return "组织名称"
    
    # 检查是否是事件
    if re.search(r'事件|运动|革命|战争|会议|活动|计划|项目|工程|改革|发展|建设|成立|开幕|闭幕|签署|协议', node_name):
        return "事件"
    
    # 检查是否是职位
    if re.search(r'主席|总理|部长|经理|总监|主任|队长|组长|负责人|董事|CEO|CTO|秘书|科长|处长|局长|厅长|院长|校长', node_name):
        return "职位"
    
    # 检查是否是金额
    if re.search(r'元|美元|欧元|英镑|日元|人民币|港元|￥|\$|€|£|¥|万元|亿元|千元', node_name):
        return "金额"
    
    # 检查是否是面积
    if re.search(r'平方米|平方公里|亩|公顷|km²|m²|平米|公里|千米', node_name):
        return "面积"
    
    # 检查是否是人数
    if re.search(r'人|名|位|口', node_name) and re.search(r'\d+', node_name):
        return "人数"
    
    # 检查是否是政策法规
    if re.search(r'法律|法规|条例|规定|政策|制度|办法|规则|标准|指南|指导|意见|通知|决定|决议', node_name):
        return "政策法规"
    
    # 检查是否是产品
    if re.search(r'产品|商品|服务|系统|平台|软件|硬件|设备|工具|应用|APP|解决方案', node_name):
        return "产品"
    
    # 检查是否是技术
    if re.search(r'技术|算法|方法|工艺|流程|专利|标准|协议|框架|架构|模型', node_name):
        return "技术"
    
    # 检查是否是行业
    if re.search(r'行业|产业|领域|市场|经济|贸易|商业|金融|制造|服务业', node_name):
        return "行业"
    
    # 检查是否是学历
    if re.search(r'博士|硕士|学士|本科|专科|高中|初中|小学|MBA|EMBA|学位', node_name):
        return "学历"
    
    # 检查是否是证书
    if re.search(r'证书|资格|执照|认证|资质|许可|文凭', node_name):
        return "证书"
    
    # 检查是否是奖项
    if re.search(r'奖|奖项|荣誉|表彰|称号', node_name):
        return "奖项"
    
    # 检查是否是联系方式
    if re.search(r'电话|邮箱|地址|网址|微信|QQ|联系人', node_name):
        return "联系方式"
    
    # 如果无法确定，根据属性字符串进一步推断
    if "time" in attributes_str.lower():
        return "时间"
    if "place" in attributes_str.lower():
        return "地理位置"
    if "person" in attributes_str.lower() or "name" in attributes_str.lower():
        return "个人姓名"
    if "organization" in attributes_str.lower() or "company" in attributes_str.lower():
        return "组织名称"
    if "event" in attributes_str.lower():
        return "事件"
    if "policy" in attributes_str.lower() or "regulation" in attributes_str.lower():
        return "政策法规"
    if "product" in attributes_str.lower():
        return "产品"
    if "technology" in attributes_str.lower() or "tech" in attributes_str.lower():
        return "技术"
    if "industry" in attributes_str.lower() or "sector" in attributes_str.lower():
        return "行业"
    
    # 如果仍然无法确定，返回默认类型
    return "事件"  # 默认为事件类型，因为它比较通用

async def _handle_relationship_record(
    parts: List[str], chunk_key: str, file_path: str
) -> Optional[Dict[str, Any]]:
    """处理关系记录"""
    if len(parts) < 6:
        return None
        
    try:
        source_entity = parts[1].strip()
        target_entity = parts[2].strip()
        relationship_description = parts[3].strip()
        relationship_keywords = parts[4].strip()
        relationship_strength = parts[5].strip().rstrip(')')
        
        try:
            strength = float(relationship_strength)
        except ValueError:
            strength = 5.0  # 默认强度
            
        return {
            "source": source_entity,
            "target": target_entity,
            "description": relationship_description,
            "keywords": relationship_keywords,
            "weight": strength,
            "source_id": chunk_key,
            "file_path": file_path,
        }
    except Exception as e:
        logger.warning(f"处理关系记录时出错: {e}, 记录: {parts}")
        return None

async def process_and_store_entities_relationships(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    force_llm_summary_on_merge: int = 6,
    llm_model_func: Optional[callable] = None,
    llm_model_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    处理并存储提取的实体和关系
    
    Args:
        entities: 实体列表
        relationships: 关系列表
        knowledge_graph_inst: 知识图谱存储
        entity_vdb: 实体向量存储
        relationships_vdb: 关系向量存储
        force_llm_summary_on_merge: 强制使用LLM合并的阈值
        llm_model_func: LLM模型函数，用于合并描述
        llm_model_kwargs: LLM模型参数
    """
    # 1. 按实体名称分组
    entity_groups = defaultdict(list)
    for entity in entities:
        entity_groups[entity["name"]].append(entity)
    
    # 2. 合并并存储实体
    entity_progress = create_progress_bar(len(entity_groups), "处理实体")
    entity_data_for_vdb = {}
    
    for i, (entity_name, entity_group) in enumerate(entity_groups.items()):
        entity_progress.update(i+1, f"处理实体: {entity_name}")
        
        # 合并实体数据
        merged_entity = await merge_entity_data(
            entity_name, 
            entity_group, 
            knowledge_graph_inst,
            force_llm_summary_on_merge,
            llm_model_func,
            llm_model_kwargs
        )
        
        # 存储到知识图谱
        await knowledge_graph_inst.upsert_node(entity_name, merged_entity)
        
        # 准备向量存储数据
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        entity_data_for_vdb[entity_id] = {
            "entity_name": entity_name,
            "entity_type": merged_entity["entity_type"],
            "content": f"{entity_name}\n{merged_entity['description']}",
            "source_id": merged_entity["source_id"],
            "file_path": merged_entity["file_path"],
        }
    
    entity_progress.finish()
    
    # 3. 按关系对分组
    relationship_groups = defaultdict(list)
    for rel in relationships:
        rel_key = (rel["source"], rel["target"])
        relationship_groups[rel_key].append(rel)
    
    # 4. 合并并存储关系
    rel_progress = create_progress_bar(len(relationship_groups), "处理关系")
    relationship_data_for_vdb = {}
    
    for i, (rel_key, rel_group) in enumerate(relationship_groups.items()):
        src_id, tgt_id = rel_key
        rel_progress.update(i+1, f"处理关系: {src_id} -> {tgt_id}")
        
        # 确保源实体和目标实体存在
        for node_id in [src_id, tgt_id]:
            if not await knowledge_graph_inst.has_node(node_id):
                # 如果实体不存在，创建一个基本实体
                entity_id = compute_mdhash_id(node_id, prefix="ent-")
                entity_name = node_id.strip('"')  # 移除引号以获取实际名称
                
                # 推断实体类型
                inferred_type = _infer_entity_type(entity_name, "")
                
                # 根据关系描述生成更好的实体描述
                entity_description = "自动创建的实体"
                for rel in rel_group:
                    if rel["source"] == node_id and "description" in rel:
                        target_name = rel['target'].replace('"', '')
                        entity_description = f"与{target_name}有关系：{rel['description']}"
                        break
                    elif rel["target"] == node_id and "description" in rel:
                        source_name = rel['source'].replace('"', '')
                        entity_description = f"与{source_name}有关系：{rel['description']}"
                        break
                
                await knowledge_graph_inst.upsert_node(
                    node_id,
                    {
                        "id": entity_id,
                        "name": entity_name,
                        "type": inferred_type,
                        "description": entity_description,
                        "entity_id": node_id,  # 保留原有字段以兼容现有代码
                        "entity_type": inferred_type,  # 保留原有字段以兼容现有代码
                        "source_id": rel_group[0]["source_id"],
                        "file_path": rel_group[0]["file_path"],
                    }
                )
        
        # 合并关系数据
        merged_relation = await merge_relationship_data(
            src_id,
            tgt_id,
            rel_group,
            knowledge_graph_inst,
            force_llm_summary_on_merge,
            llm_model_func,
            llm_model_kwargs
        )
        
        # 存储到知识图谱
        await knowledge_graph_inst.upsert_edge(
            src_id,
            tgt_id,
            {
                "weight": merged_relation["weight"],
                "description": merged_relation["description"],
                "keywords": merged_relation["keywords"],
                "source_id": merged_relation["source_id"],
                "file_path": merged_relation["file_path"],
            }
        )
        
        # 准备向量存储数据
        rel_id = compute_mdhash_id(f"{src_id}{tgt_id}", prefix="rel-")
        relationship_data_for_vdb[rel_id] = {
            "src_id": src_id,
            "tgt_id": tgt_id,
            "content": f"{src_id}\t{tgt_id}\n{merged_relation['keywords']}\n{merged_relation['description']}",
            "keywords": merged_relation["keywords"],
            "description": merged_relation["description"],
            "source_id": merged_relation["source_id"],
            "weight": merged_relation["weight"],
            "file_path": merged_relation["file_path"],
        }
    
    rel_progress.finish()
    
    # 5. 批量存储到向量数据库
    logger.info(f"存储 {len(entity_data_for_vdb)} 个实体到向量数据库")
    if entity_data_for_vdb:
        await entity_vdb.upsert(entity_data_for_vdb)
        
    logger.info(f"存储 {len(relationship_data_for_vdb)} 个关系到向量数据库")
    if relationship_data_for_vdb:
        await relationships_vdb.upsert(relationship_data_for_vdb)

async def merge_entity_data(
    entity_name: str,
    entities: List[Dict[str, Any]],
    knowledge_graph_inst: BaseGraphStorage,
    force_llm_summary_on_merge: int = 6,
    llm_model_func: Optional[callable] = None,
    llm_model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    合并实体数据
    
    Args:
        entity_name: 实体名称
        entities: 实体列表
        knowledge_graph_inst: 知识图谱存储
        force_llm_summary_on_merge: 强制使用LLM合并的阈值
        llm_model_func: LLM模型函数
        llm_model_kwargs: LLM模型参数
        
    Returns:
        合并后的实体数据
    """
    # 1. 获取已有的实体数据
    existing_entity_types = []
    existing_source_ids = []
    existing_descriptions = []
    existing_file_paths = []
    existing_id = None
    existing_attributes = {}
    
    existing_node = await knowledge_graph_inst.get_node(entity_name)
    if existing_node is not None:
        existing_entity_types.append(existing_node.get("entity_type", "UNKNOWN"))
        existing_id = existing_node.get("id", None)
        
        if "source_id" in existing_node:
            existing_source_ids.extend(
                split_string_by_multi_markers(existing_node["source_id"], [GRAPH_FIELD_SEP])
            )
            
        if "description" in existing_node:
            existing_descriptions.append(existing_node["description"])
            
        if "file_path" in existing_node:
            existing_file_paths.extend(
                split_string_by_multi_markers(existing_node["file_path"], [GRAPH_FIELD_SEP])
            )
            
        # 提取现有实体的属性
        for key, value in existing_node.items():
            if key not in ["id", "name", "type", "description", "entity_id", "entity_type", "source_id", "file_path"]:
                existing_attributes[key] = value
    
    # 2. 合并实体类型（选择出现次数最多的）
    all_entity_types = [entity.get("type", entity.get("entity_type", "UNKNOWN")) for entity in entities] + existing_entity_types
    entity_type_counts = {}
    for et in all_entity_types:
        if et and et != "UNKNOWN":
            entity_type_counts[et] = entity_type_counts.get(et, 0) + 1
    
    # 如果有有效的实体类型，选择最常见的；否则尝试推断
    if entity_type_counts:
        entity_type = max(entity_type_counts.items(), key=lambda x: x[1])[0]
    else:
        # 如果没有有效的实体类型，尝试从实体名称推断
        entity_type = await _infer_entity_type(entity_name, "")
    
    # 3. 合并描述 - 改进描述合并逻辑
    all_descriptions = []
    for entity in entities:
        if "description" in entity and entity["description"].strip():
            # 清理描述文本，移除重复信息
            desc = clean_str(entity["description"])
            if desc not in all_descriptions:
                all_descriptions.append(desc)
    
    # 添加现有描述
    for desc in existing_descriptions:
        if desc.strip() and desc not in all_descriptions:
            all_descriptions.append(desc)
    
    # 4. 合并源ID和文件路径
    all_source_ids = [entity["source_id"] for entity in entities if "source_id" in entity] + existing_source_ids
    all_file_paths = [entity.get("file_path", "unknown_source") for entity in entities] + existing_file_paths
    
    source_id = GRAPH_FIELD_SEP.join(set(all_source_ids))
    file_path = GRAPH_FIELD_SEP.join(set(all_file_paths))
    
    # 5. 合并其他属性
    merged_attributes = existing_attributes.copy()
    for entity in entities:
        for key, value in entity.items():
            if key not in ["id", "name", "type", "description", "entity_id", "entity_type", "source_id", "file_path"]:
                # 如果是新属性，直接添加；如果是已有属性，合并值
                if key in merged_attributes and isinstance(merged_attributes[key], str) and isinstance(value, str):
                    # 对于字符串类型的属性，避免重复
                    if value not in merged_attributes[key]:
                        merged_attributes[key] = f"{merged_attributes[key]}{GRAPH_FIELD_SEP}{value}"
                else:
                    merged_attributes[key] = value
    
    # 6. 使用LLM合并描述 - 改进摘要处理
    description = ""
    num_fragments = len(all_descriptions)
    
    # 如果只有一个描述，直接使用
    if num_fragments == 1:
        description = all_descriptions[0]
    # 如果描述数量超过阈值，使用LLM合并
    elif num_fragments >= force_llm_summary_on_merge and llm_model_func:
        logger.info(f"使用LLM合并实体 {entity_name} 的 {num_fragments} 个描述片段")
        
        # 使用LLM合并描述，提供更多上下文
        summary_prompt = PROMPTS["summarize_entity_descriptions"].format(
            entity_name=entity_name,
            description_list="\n- " + "\n- ".join(all_descriptions),
        )
        
        try:
            summary = await llm_model_func(summary_prompt, **llm_model_kwargs)
            description = summary.strip()
            
            # 确保摘要包含实体名称
            if entity_name not in description:
                description = f"{entity_name}：{description}"
        except Exception as e:
            logger.error(f"使用LLM合并实体描述时出错: {e}")
            # 失败时使用简单连接
            description = "\n".join(all_descriptions)
    else:
        # 简单连接所有描述
        description = "\n".join(all_descriptions)
    
    # 生成或使用现有的ID
    entity_id = existing_id if existing_id else compute_mdhash_id(entity_name, prefix="ent-")
    
    # 构建最终的实体数据
    result = {
        "id": entity_id,
        "name": entity_name,
        "type": entity_type,
        "description": description,
        "entity_id": entity_name,  # 保留原有字段以兼容现有代码
        "entity_type": entity_type,  # 保留原有字段以兼容现有代码
        "source_id": source_id,
        "file_path": file_path,
    }
    
    # 添加合并的属性
    result.update(merged_attributes)
    
    return result

async def merge_relationship_data(
    src_id: str,
    tgt_id: str,
    relationships: List[Dict[str, Any]],
    knowledge_graph_inst: BaseGraphStorage,
    force_llm_summary_on_merge: int = 6,
    llm_model_func: Optional[callable] = None,
    llm_model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    合并关系数据
    
    Args:
        src_id: 源实体ID
        tgt_id: 目标实体ID
        relationships: 关系列表
        knowledge_graph_inst: 知识图谱存储
        force_llm_summary_on_merge: 强制使用LLM合并的阈值
        llm_model_func: LLM模型函数
        llm_model_kwargs: LLM模型参数
        
    Returns:
        合并后的关系数据
    """
    # 1. 获取已有的关系数据
    existing_weights = []
    existing_source_ids = []
    existing_descriptions = []
    existing_keywords = []
    existing_file_paths = []
    existing_attributes = {}
    
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        existing_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        if existing_edge:
            existing_weights.append(existing_edge.get("weight", 1.0))
            
            if "source_id" in existing_edge:
                existing_source_ids.extend(
                    split_string_by_multi_markers(existing_edge["source_id"], [GRAPH_FIELD_SEP])
                )
                
            if "description" in existing_edge:
                existing_descriptions.append(existing_edge["description"])
                
            if "keywords" in existing_edge:
                existing_keywords.extend(
                    split_string_by_multi_markers(existing_edge["keywords"], [GRAPH_FIELD_SEP])
                )
                
            if "file_path" in existing_edge:
                existing_file_paths.extend(
                    split_string_by_multi_markers(existing_edge["file_path"], [GRAPH_FIELD_SEP])
                )
                
            # 提取现有关系的其他属性
            for key, value in existing_edge.items():
                if key not in ["weight", "description", "keywords", "source_id", "file_path"]:
                    existing_attributes[key] = value
    
    # 2. 合并权重（使用最大值而不是求和，避免权重过大）
    all_weights = [float(rel["weight"]) for rel in relationships if "weight" in rel] + existing_weights
    weight = max(all_weights) if all_weights else 1.0
    
    # 3. 合并描述 - 改进描述合并逻辑
    all_descriptions = []
    for rel in relationships:
        if "description" in rel and rel["description"].strip():
            # 清理描述文本，移除重复信息
            desc = clean_str(rel["description"])
            if desc not in all_descriptions:
                all_descriptions.append(desc)
    
    # 添加现有描述
    for desc in existing_descriptions:
        if desc.strip() and desc not in all_descriptions:
            all_descriptions.append(desc)
    
    # 4. 合并关键词 - 改进关键词合并逻辑
    all_keywords_list = []
    for rel in relationships:
        if "keywords" in rel and rel["keywords"].strip():
            # 分割关键词并清理
            keywords_parts = rel["keywords"].split(",")
            for kw in keywords_parts:
                kw = kw.strip()
                if kw and kw not in all_keywords_list:
                    all_keywords_list.append(kw)
    
    # 添加现有关键词
    for kw in existing_keywords:
        if kw.strip() and kw not in all_keywords_list:
            all_keywords_list.append(kw)
    
    # 5. 合并源ID和文件路径
    all_source_ids = [rel["source_id"] for rel in relationships if "source_id" in rel] + existing_source_ids
    all_file_paths = [rel.get("file_path", "unknown_source") for rel in relationships] + existing_file_paths
    
    source_id = GRAPH_FIELD_SEP.join(set(all_source_ids))
    file_path = GRAPH_FIELD_SEP.join(set(all_file_paths))
    
    # 6. 合并其他属性
    merged_attributes = existing_attributes.copy()
    for rel in relationships:
        for key, value in rel.items():
            if key not in ["weight", "description", "keywords", "source_id", "file_path"]:
                # 如果是新属性，直接添加；如果是已有属性，合并值
                if key in merged_attributes and isinstance(merged_attributes[key], str) and isinstance(value, str):
                    # 对于字符串类型的属性，避免重复
                    if value not in merged_attributes[key]:
                        merged_attributes[key] = f"{merged_attributes[key]}{GRAPH_FIELD_SEP}{value}"
                else:
                    merged_attributes[key] = value
    
    # 7. 使用LLM合并描述 - 改进摘要处理
    description = ""
    num_fragments = len(all_descriptions)
    
    # 获取源实体和目标实体的名称，用于更好的描述
    src_name = src_id.strip('"')
    tgt_name = tgt_id.strip('"')
    
    # 如果只有一个描述，直接使用
    if num_fragments == 1:
        description = all_descriptions[0]
    # 如果描述数量超过阈值，使用LLM合并
    elif num_fragments >= force_llm_summary_on_merge and llm_model_func:
        logger.info(f"使用LLM合并关系 {src_name} -> {tgt_name} 的 {num_fragments} 个描述片段")
        
        # 使用LLM合并描述，提供更多上下文
        summary_prompt = PROMPTS["summarize_entity_descriptions"].format(
            entity_name=f"({src_name}, {tgt_name})的关系",
            description_list="\n- " + "\n- ".join(all_descriptions),
        )
        
        try:
            summary = await llm_model_func(summary_prompt, **llm_model_kwargs)
            description = summary.strip()
            
            # 确保摘要包含实体名称
            if src_name not in description or tgt_name not in description:
                description = f"{src_name}与{tgt_name}的关系：{description}"
        except Exception as e:
            logger.error(f"使用LLM合并关系描述时出错: {e}")
            # 失败时使用简单连接
            description = "\n".join(all_descriptions)
    else:
        # 简单连接所有描述
        description = "\n".join(all_descriptions)
    
    # 合并关键词为逗号分隔的字符串
    keywords = ", ".join(all_keywords_list)
    
    # 构建最终的关系数据
    result = {
        "weight": weight,
        "description": description,
        "keywords": keywords,
        "source_id": source_id,
        "file_path": file_path,
    }
    
    # 添加合并的属性
    result.update(merged_attributes)
    
    return result
