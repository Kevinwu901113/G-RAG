import asyncio
import html
import io
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List, Optional
import xml.etree.ElementTree as ET

import numpy as np
import tiktoken

from .logger_manager import get_logger, configure_logging

ENCODER = None

# 获取grag日志器
logger = get_logger("grag")


def set_logger(log_file: str):
    """
    配置日志系统
    
    Args:
        log_file: 日志文件路径
    """
    # 提取日志目录
    log_dir = os.path.dirname(log_file)
    if not log_dir:
        log_dir = "logs"
    
    # 使用新的日志管理器
    log_manager = configure_logging(log_dir=log_dir, default_level="debug")
    
    # 创建会话日志文件
    session_log_file = log_manager.create_session_log_file(prefix="grag")
    
    # 记录系统信息
    log_manager.log_system_info()
    
    logger.info(f"日志将被保存到: {session_log_file}")
    
    return log_manager


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        return None


def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_csv(data: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> List[List[str]]:
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]


def save_data_to_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_combine_contexts(hl, ll):
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources = []
    seen = set()

    for item in list_hl + list_ll:
        if item and item not in seen:
            combined_sources.append(item)
            seen.add(item)

    combined_sources_result = [",\t".join(header)]

    for i, item in enumerate(combined_sources, start=1):
        combined_sources_result.append(f"{i},\t{item}")

    combined_sources_result = "\n".join(combined_sources_result)

    return combined_sources_result


class ProgressBar:
    """
    用于显示进度条的类，简化版本，避免使用ANSI转义序列
    """
    def __init__(self, total: int, prefix: str = '', suffix: str = '', 
                 decimals: int = 1, length: int = 50, fill: str = '█', 
                 print_end: str = '\r', log_file: Optional[str] = None):
        """
        初始化进度条
        
        Args:
            total: 总迭代次数
            prefix: 前缀字符串
            suffix: 后缀字符串
            decimals: 百分比的小数位数
            length: 进度条的字符长度
            fill: 进度条填充字符
            print_end: 打印结束字符
            log_file: 日志文件路径，如果提供则同时记录到日志文件
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.log_file = log_file
        self.iteration = 0
        self.start_time = time.time()
        self._print_progress()
    
    def update(self, iteration: Optional[int] = None, suffix: Optional[str] = None):
        """
        更新进度条
        
        Args:
            iteration: 当前迭代次数，如果为None则自动加1
            suffix: 更新后缀文本
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
            
        if suffix is not None:
            self.suffix = suffix
            
        self._print_progress()
    
    def _print_progress(self):
        """打印进度条"""
        if self.total <= 0:
            percent = "100.0"
            filled_length = self.length
        else:
            percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
            filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        
        # 计算已用时间和估计剩余时间
        elapsed_time = time.time() - self.start_time
        if self.iteration > 0:
            eta = elapsed_time * (self.total / self.iteration - 1)
            time_info = f" | 用时: {self._format_time(elapsed_time)} | 剩余: {self._format_time(eta)}"
        else:
            time_info = ""
            
        progress_line = f'\r{self.prefix} |{bar}| {percent}% {self.suffix}{time_info}'
        
        # 打印到控制台
        sys.stdout.write(progress_line + self.print_end)
        sys.stdout.flush()
        
        # 如果提供了日志文件，也记录到日志
        if self.log_file and logger.handlers:
            logger.info(progress_line.strip())
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间为人类可读格式"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
    
    def finish(self):
        """完成进度条"""
        self.update(self.total)
        # 打印换行符，使下一行输出从新行开始
        sys.stdout.write('\n')
        sys.stdout.flush()
        
        # 记录完成信息到日志
        if self.log_file and logger.handlers:
            elapsed_time = time.time() - self.start_time
            logger.info(f"任务完成，总用时: {self._format_time(elapsed_time)}")


def create_progress_bar(total: int, description: str, log_file: Optional[str] = None) -> ProgressBar:
    """
    创建并返回一个进度条实例
    
    Args:
        total: 总迭代次数
        description: 进度条描述
        log_file: 日志文件路径
        
    Returns:
        ProgressBar实例
    """
    return ProgressBar(
        total=total,
        prefix=f'{description}',
        suffix='完成',
        length=50,
        log_file=log_file
    )
