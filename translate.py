#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import asyncio
import argparse
import time
import re
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

import ollama
import tqdm
from openai import AsyncOpenAI

from grag.utils.common import logger


class TranslationStrategy:
    """翻译策略基类"""
    
    async def translate(self, text: str, system_prompt: str, max_retries: int, timeout: int) -> str:
        """执行翻译
        
        Args:
            text: 待翻译文本
            system_prompt: 系统提示
            max_retries: 最大重试次数
            timeout: 超时时间
            
        Returns:
            翻译后的文本
        """
        raise NotImplementedError("子类必须实现此方法")


class OllamaTranslationStrategy(TranslationStrategy):
    """使用Ollama进行翻译的策略"""
    
    def __init__(self, model_name: str, host: str):
        """初始化Ollama翻译策略
        
        Args:
            model_name: Ollama模型名称
            host: Ollama服务器地址
        """
        self.model_name = model_name
        self.host = host
        self.ollama_client = ollama.AsyncClient(host=host)
    
    async def translate(self, text: str, system_prompt: str, max_retries: int, timeout: int) -> str:
        """使用Ollama模型翻译文本
        
        Args:
            text: 待翻译文本
            system_prompt: 系统提示
            max_retries: 最大重试次数
            timeout: 超时时间
            
        Returns:
            翻译后的文本
        """
        for retry in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.ollama_client.chat(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text}
                        ],
                        options={"temperature": 0.1}
                    ),
                    timeout=timeout
                )
                return response["message"]["content"]
            except Exception as e:
                logger.warning(f"Ollama翻译失败（第{retry+1}次尝试）: {str(e)}")
                if retry < max_retries - 1:
                    # 指数退避重试
                    await asyncio.sleep(2 ** retry)
                else:
                    logger.error(f"Ollama翻译失败，已达到最大重试次数: {text[:100]}...")
                    return f"[翻译失败] {text}"


class OpenAITranslationStrategy(TranslationStrategy):
    """使用OpenAI API进行翻译的策略"""
    
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, max_concurrency: int = 5):
        """初始化OpenAI翻译策略
        
        Args:
            model_name: OpenAI模型名称
            api_key: OpenAI API密钥
            base_url: 自定义API基础URL
            max_concurrency: 最大并发请求数量
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
        # 翻译缓存，用于避免重复翻译相同的文本
        self.translation_cache = {}
        
        # 设置API密钥
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    async def _translate_single(self, text: str, system_prompt: str, max_retries: int, timeout: int) -> str:
        """翻译单个文本
        
        Args:
            text: 待翻译文本
            system_prompt: 系统提示
            max_retries: 最大重试次数
            timeout: 超时时间
            
        Returns:
            翻译后的文本
        """
        # 生成缓存键（使用文本和系统提示的组合）
        cache_key = f"{text}:{system_prompt}"
        
        # 检查缓存中是否已有翻译结果
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
            
        # 创建OpenAI客户端
        openai_client = AsyncOpenAI() if self.base_url is None else AsyncOpenAI(base_url=self.base_url)
        
        for retry in range(max_retries):
            try:
                async with self.semaphore:  # 使用信号量控制并发
                    response = await asyncio.wait_for(
                        openai_client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": text}
                            ],
                            temperature=0.1
                        ),
                        timeout=timeout
                    )
                result = response.choices[0].message.content
                # 将结果存入缓存
                self.translation_cache[cache_key] = result
                return result
            except Exception as e:
                logger.warning(f"OpenAI翻译失败（第{retry+1}次尝试）: {str(e)}")
                if retry < max_retries - 1:
                    # 指数退避重试
                    await asyncio.sleep(2 ** retry)
                else:
                    logger.error(f"OpenAI翻译失败，已达到最大重试次数: {text[:100]}...")
                    return f"[翻译失败] {text}"
    
    async def translate(self, text: str, system_prompt: str, max_retries: int, timeout: int) -> str:
        """使用OpenAI API翻译文本
        
        Args:
            text: 待翻译文本
            system_prompt: 系统提示
            max_retries: 最大重试次数
            timeout: 超时时间
            
        Returns:
            翻译后的文本
        """
        return await self._translate_single(text, system_prompt, max_retries, timeout)
    
    async def translate_batch(self, texts: List[str], system_prompt: str, max_retries: int, timeout: int) -> List[str]:
        """批量翻译多个文本
        
        Args:
            texts: 待翻译的文本列表
            system_prompt: 系统提示
            max_retries: 最大重试次数
            timeout: 超时时间
            
        Returns:
            翻译后的文本列表
        """
        # 检查缓存，只对未缓存的文本创建任务
        results = [None] * len(texts)
        tasks = []
        task_indices = []
        
        for i, text in enumerate(texts):
            cache_key = f"{text}:{system_prompt}"
            if cache_key in self.translation_cache:
                # 如果在缓存中找到，直接使用缓存结果
                results[i] = self.translation_cache[cache_key]
            else:
                # 否则创建翻译任务
                tasks.append(self._translate_single(text, system_prompt, max_retries, timeout))
                task_indices.append(i)
        
        # 如果有需要翻译的文本，执行批量翻译
        if tasks:
            # 使用gather并发执行所有任务
            task_results = await asyncio.gather(*tasks)
            
            # 将结果放回对应位置
            for task_idx, result in zip(task_indices, task_results):
                results[task_idx] = result
        
        return results


class HotpotQATranslator:
    """HotpotQA数据集翻译器，将英文数据集翻译成中文"""

    def __init__(
        self,
        input_dir: str = "data",
        output_dir: str = "data_translate",
        model_name: str = "qwen2.5:7b-instruct-fp16",
        host: str = "http://localhost:11434",
        max_retries: int = 3,
        timeout: int = 60,
        batch_size: int = 1,
        openai_model: str = "gpt-3.5-turbo",
        openai_api_key: str = None,
        openai_base_url: str = None,
        max_concurrency: int = 5,
        cache_file: str = "translation_cache.json",
    ):
        """初始化翻译器

        Args:
            input_dir: 输入数据目录
            output_dir: 输出数据目录
            model_name: Ollama模型名称
            host: Ollama服务器地址
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            batch_size: 批处理大小（每次写入文件的条目数）
            openai_model: OpenAI模型名称
            openai_api_key: OpenAI API密钥
            openai_base_url: OpenAI API基础URL
            max_concurrency: OpenAI API最大并发请求数量
            cache_file: 翻译缓存文件路径
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.host = host
        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = batch_size
        self.cache_file = Path(cache_file)
        
        # 初始化翻译策略
        self.ollama_strategy = OllamaTranslationStrategy(model_name, host)
        self.openai_strategy = OpenAITranslationStrategy(openai_model, openai_api_key, openai_base_url, max_concurrency)

        # 确保输出目录存在
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 加载翻译缓存
        self._load_translation_cache()

    def is_chinese_text(self, text: str) -> bool:
        """检测文本是否为中文

        Args:
            text: 待检测的文本

        Returns:
            是否为中文文本
        """
        if not text or len(text) < 5:  # 文本太短不做检测
            return True
            
        # 检测ASCII字符比例
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        ascii_ratio = ascii_chars / len(text)
        
        # 如果ASCII字符比例过高，可能是英文
        if ascii_ratio > 0.7:  # 70%以上是ASCII字符，可能是英文
            return False
            
        # 检测英文单词比例
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        if len(english_words) > 5 and len(english_words) / len(text.split()) > 0.5:
            return False
            
        return True

    def _should_use_openai(self) -> bool:
        """判断当前是否应该使用OpenAI API进行翻译
        
        在凌晨00:30到早上08:30期间使用OpenAI API，其他时间使用Ollama
        
        Returns:
            是否应该使用OpenAI API
        """
        now = datetime.datetime.now().time()
        start_time = datetime.time(0, 30)  # 00:30
        end_time = datetime.time(0, 30)    # 08:30
        
        # 判断当前时间是否在指定范围内
        if start_time <= now <= end_time:
            return True
        return False
    
    def _get_current_strategy(self) -> TranslationStrategy:
        """获取当前应该使用的翻译策略
        
        Returns:
            翻译策略实例
        """
        if self._should_use_openai():
            logger.info("当前时间在00:30-08:30范围内，使用OpenAI API进行翻译")
            return self.openai_strategy
        else:
            logger.info("当前时间不在00:30-08:30范围内，使用Ollama进行翻译")
            return self.ollama_strategy
    
    async def translate_text(self, text: str, max_chinese_retries: int = 2) -> str:
        """翻译文本，根据时间自动选择翻译策略

        Args:
            text: 待翻译的英文文本
            max_chinese_retries: 中文检测失败后的最大重试次数

        Returns:
            翻译后的中文文本
        """
        # 检查缓存中是否已有翻译结果
        if text in self._translation_cache:
            return self._translation_cache[text]
            
        system_prompt = "你是一个专业的英译中翻译助手，请将以下英文文本翻译成中文，保持原文的意思和风格，只返回翻译结果，不要添加任何解释或额外内容。"
        
        # 获取当前应该使用的翻译策略
        strategy = self._get_current_strategy()
        
        # 执行翻译
        translated_text = await strategy.translate(text, system_prompt, self.max_retries, self.timeout)
        
        # 检测翻译结果是否为中文
        chinese_retry = 0
        while not self.is_chinese_text(translated_text) and chinese_retry < max_chinese_retries:
            logger.warning(f"翻译结果可能不是中文，重试翻译: {translated_text[:50]}...")
            chinese_retry += 1
            # 加强翻译提示
            enhanced_prompt = "你是一个专业的英译中翻译助手。请将以下英文文本翻译成中文，必须使用中文字符，不要保留英文单词。保持原文的意思和风格，只返回翻译结果。"
            
            # 再次使用当前策略进行翻译
            translated_text = await strategy.translate(text, enhanced_prompt, self.max_retries, self.timeout)
        
        # 将结果添加到缓存
        self._translation_cache[text] = translated_text
        
        return translated_text
        
    # 翻译缓存字典
    _translation_cache = {}
    
    def _load_translation_cache(self):
        """从文件加载翻译缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._translation_cache = json.load(f)
                logger.info(f"已从{self.cache_file}加载{len(self._translation_cache)}条翻译缓存")
            except Exception as e:
                logger.warning(f"加载翻译缓存失败: {str(e)}")
                self._translation_cache = {}
        else:
            logger.info(f"未找到缓存文件{self.cache_file}，将创建新的翻译缓存")
            self._translation_cache = {}
    
    def _save_translation_cache(self):
        """将翻译缓存保存到文件"""
        try:
            # 限制缓存大小，避免文件过大
            if len(self._translation_cache) > 100000:
                logger.warning(f"翻译缓存过大({len(self._translation_cache)}条)，将只保留最新的100000条")
                # 转换为列表，保留最新的10万条
                cache_items = list(self._translation_cache.items())
                self._translation_cache = dict(cache_items[-100000:])
            
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._translation_cache, f, ensure_ascii=False)
            logger.info(f"已将{len(self._translation_cache)}条翻译缓存保存到{self.cache_file}")
        except Exception as e:
            logger.error(f"保存翻译缓存失败: {str(e)}")
    
    async def translate_texts_batch(self, texts: List[str], max_chinese_retries: int = 2) -> List[str]:
        """批量翻译多个文本，根据时间自动选择翻译策略

        Args:
            texts: 待翻译的英文文本列表
            max_chinese_retries: 中文检测失败后的最大重试次数

        Returns:
            翻译后的中文文本列表
        """
        if not texts:
            return []
        
        # 检查全局翻译缓存
        results = [None] * len(texts)
        texts_to_translate = []
        text_indices = []
        
        # 先检查缓存
        for i, text in enumerate(texts):
            if text in self._translation_cache:
                results[i] = self._translation_cache[text]
            else:
                texts_to_translate.append(text)
                text_indices.append(i)
        
        # 如果所有文本都在缓存中，直接返回结果
        if not texts_to_translate:
            return results
            
        system_prompt = "你是一个专业的英译中翻译助手，请将以下英文文本翻译成中文，保持原文的意思和风格，只返回翻译结果，不要添加任何解释或额外内容。"
        
        # 获取当前应该使用的翻译策略
        strategy = self._get_current_strategy()
        
        # 如果是OpenAI策略且支持批量翻译
        if isinstance(strategy, OpenAITranslationStrategy) and hasattr(strategy, 'translate_batch'):
            # 执行批量翻译
            translated_texts = await strategy.translate_batch(texts_to_translate, system_prompt, self.max_retries, self.timeout)
            
            # 检查每个翻译结果是否为中文，如果不是则单独重试
            for i, translated_text in enumerate(translated_texts):
                chinese_retry = 0
                while not self.is_chinese_text(translated_text) and chinese_retry < max_chinese_retries:
                    logger.warning(f"批量翻译结果中第{i+1}个可能不是中文，单独重试翻译: {translated_text[:50]}...")
                    chinese_retry += 1
                    # 加强翻译提示
                    enhanced_prompt = "你是一个专业的英译中翻译助手。请将以下英文文本翻译成中文，必须使用中文字符，不要保留英文单词。保持原文的意思和风格，只返回翻译结果。"
                    
                    # 单独重试这一个文本
                    translated_texts[i] = await strategy.translate(texts_to_translate[i], enhanced_prompt, self.max_retries, self.timeout)
                
                # 将结果添加到缓存
                self._translation_cache[texts_to_translate[i]] = translated_texts[i]
            
            # 将翻译结果放回原位置
            for idx, result in zip(text_indices, translated_texts):
                results[idx] = result
                
            return results
        else:
            # 如果不支持批量翻译，则单独翻译每个文本
            translated_texts = []
            for text in texts_to_translate:
                translated_text = await self.translate_text(text, max_chinese_retries)
                # 添加到缓存
                self._translation_cache[text] = translated_text
                translated_texts.append(translated_text)
            
            # 将翻译结果放回原位置
            for idx, result in zip(text_indices, translated_texts):
                results[idx] = result
                
            return results

    async def translate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """翻译单个HotpotQA条目

        Args:
            item: HotpotQA数据条目

        Returns:
            翻译后的数据条目
        """
        translated_item = item.copy()

        # 收集所有需要翻译的文本，包括问题、答案和上下文
        all_texts_to_translate = []
        text_mapping = {}  # 用于记录文本在原始数据中的位置
        
        # 收集问题和答案
        if "question" in item and item["question"]:
            text_id = len(all_texts_to_translate)
            all_texts_to_translate.append(item["question"])
            text_mapping[text_id] = ("question", None, None)
            translated_item["question_en"] = item["question"]  # 保留英文原文
            
        if "answer" in item and item["answer"]:
            text_id = len(all_texts_to_translate)
            all_texts_to_translate.append(item["answer"])
            text_mapping[text_id] = ("answer", None, None)
            translated_item["answer_en"] = item["answer"]  # 保留英文原文
        
        # 收集上下文中的标题和句子
        if "context" in item and isinstance(item["context"], list):
            translated_item["context_en"] = item["context"]  # 保留英文原文
            translated_context = []
            
            for ctx_idx, title_sentences in enumerate(item["context"]):
                if len(title_sentences) == 2:
                    title = title_sentences[0]
                    sentences = title_sentences[1]
                    
                    # 添加标题到翻译列表
                    text_id = len(all_texts_to_translate)
                    all_texts_to_translate.append(title)
                    text_mapping[text_id] = ("context_title", ctx_idx, None)
                    
                    # 添加句子到翻译列表（过滤空句子）
                    for sent_idx, sentence in enumerate(sentences):
                        if sentence.strip():
                            text_id = len(all_texts_to_translate)
                            all_texts_to_translate.append(sentence)
                            text_mapping[text_id] = ("context_sentence", ctx_idx, sent_idx)
                    
                    # 为这个上下文项创建占位符
                    translated_context.append([None, [None] * len(sentences)])
                else:
                    # 保持原样
                    translated_context.append(title_sentences)
            
            translated_item["context"] = translated_context
        
        # 如果有文本需要翻译，则批量翻译
        if all_texts_to_translate:
            # 使用批量翻译方法一次性翻译所有文本
            all_translated_texts = await self.translate_texts_batch(all_texts_to_translate)
            
            # 将翻译结果放回原位置
            for text_id, translated_text in enumerate(all_translated_texts):
                text_type, ctx_idx, sent_idx = text_mapping[text_id]
                
                if text_type == "question":
                    translated_item["question"] = translated_text
                elif text_type == "answer":
                    translated_item["answer"] = translated_text
                elif text_type == "context_title":
                    translated_item["context"][ctx_idx][0] = translated_text
                elif text_type == "context_sentence":
                    translated_item["context"][ctx_idx][1][sent_idx] = translated_text

        return translated_item

    def get_json_files(self) -> List[Path]:
        """获取输入目录中的所有JSON文件

        Returns:
            JSON文件路径列表
        """
        return list(self.input_dir.glob("**/*.json"))

    def get_translated_items(self, output_file: Path) -> Dict[str, Any]:
        """获取已翻译的条目，用于断点续传

        Args:
            output_file: 输出文件路径

        Returns:
            已翻译的条目字典，键为条目ID
        """
        if not output_file.exists():
            return {}

        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 创建ID到条目的映射，使用问题作为ID
                return {item.get("_id", item.get("question", "")): item for item in data}
        except json.JSONDecodeError:
            logger.warning(f"输出文件 {output_file} 格式错误，将重新创建")
            return {}
        except Exception as e:
            logger.error(f"读取输出文件 {output_file} 失败: {str(e)}")
            return {}

    async def translate_file(self, input_file: Path) -> None:
        """翻译单个JSON文件

        Args:
            input_file: 输入文件路径
        """
        # 确定输出文件路径
        rel_path = input_file.relative_to(self.input_dir)
        output_file = self.output_dir / rel_path
        output_file.parent.mkdir(exist_ok=True, parents=True)

        logger.info(f"开始翻译文件: {input_file} -> {output_file}")

        try:
            # 读取输入文件
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 获取已翻译的条目
            translated_items = self.get_translated_items(output_file)
            logger.info(f"已找到 {len(translated_items)} 个已翻译条目")

            # 创建进度条
            progress_bar = tqdm.tqdm(
                total=len(data),
                desc=f"翻译 {input_file.name}",
                initial=len(translated_items),
            )

            # 批量处理和保存
            batch = []
            batch_count = 0

            # 遍历数据条目
            for item in data:
                # 使用ID或问题作为唯一标识
                item_id = item.get("_id", item.get("question", ""))
                
                # 如果已翻译，则跳过
                if item_id in translated_items:
                    continue

                # 翻译条目
                translated_item = await self.translate_item(item)
                batch.append(translated_item)
                batch_count += 1
                progress_bar.update(1)

                # 达到批处理大小时保存
                if batch_count >= self.batch_size:
                    await self.save_batch(output_file, batch, translated_items)
                    batch = []
                    batch_count = 0

            # 保存剩余的批次
            if batch:
                await self.save_batch(output_file, batch, translated_items)

            progress_bar.close()
            logger.info(f"文件 {input_file.name} 翻译完成")

        except Exception as e:
            logger.error(f"翻译文件 {input_file} 失败: {str(e)}")

    def validate_json_structure(self, original_data: List[Dict[str, Any]], translated_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """验证翻译后的JSON结构与原始数据结构是否一致

        Args:
            original_data: 原始数据
            translated_data: 翻译后的数据

        Returns:
            验证结果和错误信息
        """
        # 注意：在增量翻译场景下，translated_data的数量可能小于original_data
        # 因此不再检查数据条目数量是否完全一致
            
        # 检查第一条数据的字段结构
        if not original_data or not translated_data:
            return True, ""
            
        # 获取原始数据的字段结构
        sample_original = original_data[0]
        sample_translated = translated_data[0]
        
        # 检查原始字段是否都存在于翻译后的数据中
        for key in sample_original.keys():
            if key not in sample_translated and key + "_en" not in sample_translated:
                return False, f"翻译数据缺少原始字段: {key}"
                
        # 检查嵌套结构（以context为例）
        if "context" in sample_original and "context" in sample_translated:
            if not isinstance(sample_original["context"], type(sample_translated["context"])):
                return False, f"context字段类型不一致: 原始类型 {type(sample_original['context']).__name__}，翻译类型 {type(sample_translated['context']).__name__}"
                
            # 如果是列表，检查列表结构
            if isinstance(sample_original["context"], list) and sample_original["context"] and sample_translated["context"]:
                if len(sample_original["context"]) > 0 and len(sample_translated["context"]) > 0:
                    # 检查第一个元素的结构
                    if not isinstance(sample_original["context"][0], type(sample_translated["context"][0])):
                        return False, f"context列表元素类型不一致"
        
        return True, ""

    async def save_batch(self, output_file: Path, batch: List[Dict[str, Any]], 
                         translated_items: Dict[str, Any]) -> None:
        """保存批量翻译结果

        Args:
            output_file: 输出文件路径
            batch: 待保存的批量数据
            translated_items: 已翻译的条目字典，用于更新
        """
        if not batch:
            return

        # 更新已翻译条目字典
        for item in batch:
            item_id = item.get("_id", item.get("question", ""))
            translated_items[item_id] = item

        # 将字典转换为列表
        all_items = list(translated_items.values())
        
        # 读取原始文件以验证结构
        try:
            original_file = output_file.relative_to(self.output_dir)
            original_file = self.input_dir / original_file
            with open(original_file, "r", encoding="utf-8") as f:
                original_data = json.load(f)
                
            # 验证JSON结构 - 只在有数据时进行验证
            if all_items:
                is_valid, error_msg = self.validate_json_structure(original_data, all_items)
                if not is_valid:
                    logger.error(f"JSON结构验证失败: {error_msg}")
                    logger.warning(f"原始数据条目数: {len(original_data)}, 翻译数据条目数: {len(all_items)}")
                    logger.warning("保存操作将继续，但请检查翻译逻辑以确保数据完整性")
                    # 不再返回，继续执行保存操作
            else:
                logger.warning("没有翻译数据可保存")
                return
        except Exception as e:
            logger.warning(f"无法验证JSON结构: {str(e)}，将继续保存")

        # 保存到文件
        try:
            # 先保存到临时文件，验证JSON完整性
            temp_file = output_file.with_suffix(".tmp.json")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(all_items, f, ensure_ascii=False, indent=2)
                
            # 验证JSON文件完整性
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    json.load(f)  # 尝试加载JSON以验证完整性
                    
                # 验证成功，重命名为正式文件
                if output_file.exists():
                    output_file.unlink()  # 删除旧文件
                temp_file.rename(output_file)
            except json.JSONDecodeError as e:
                logger.error(f"生成的JSON文件格式错误: {str(e)}")
                if temp_file.exists():
                    temp_file.unlink()  # 删除临时文件
        except Exception as e:
            logger.error(f"保存翻译结果到 {output_file} 失败: {str(e)}")

    async def translate_all(self) -> None:
        """翻译所有JSON文件"""
        json_files = self.get_json_files()
        logger.info(f"找到 {len(json_files)} 个JSON文件需要翻译")

        for json_file in json_files:
            await self.translate_file(json_file)
            # 每翻译完一个文件就保存一次缓存
            self._save_translation_cache()
            
        # 翻译完成后再次保存缓存
        self._save_translation_cache()


async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="将HotpotQA英文数据集翻译成中文")
    parser.add_argument("--input", "-i", default="data", help="输入数据目录")
    parser.add_argument("--output", "-o", default="data_translate", help="输出数据目录")
    parser.add_argument("--model", "-m", default="qwen2.5:7b-instruct-fp16", help="Ollama模型名称")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama服务器地址")
    parser.add_argument("--retries", "-r", type=int, default=3, help="最大重试次数")
    parser.add_argument("--timeout", "-t", type=int, default=60, help="请求超时时间（秒）")
    parser.add_argument("--batch", "-b", type=int, default=100, help="批处理大小")
    parser.add_argument("--openai-model", default="deepseek-chat", help="OpenAI模型名称")
    parser.add_argument("--openai-api-key", default="sk-f3333bd9b06a4fa6b1520f06c010c736", help="OpenAI API密钥")
    parser.add_argument("--openai-base-url", default="https://api.deepseek.com", help="OpenAI API基础URL")
    parser.add_argument("--max-concurrency", type=int, default=100, help="OpenAI API最大并发请求数量")
    parser.add_argument("--cache-file", default="translation_cache.json", help="翻译缓存文件路径")
    args = parser.parse_args()

    # 创建翻译器
    translator = HotpotQATranslator(
        input_dir=args.input,
        output_dir=args.output,
        model_name=args.model,
        host=args.host,
        max_retries=args.retries,
        timeout=args.timeout,
        batch_size=args.batch,
        openai_model=args.openai_model,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        max_concurrency=args.max_concurrency,
    )

    # 开始翻译
    logger.info(f"开始翻译HotpotQA数据集，从 {args.input} 到 {args.output}")
    start_time = time.time()
    await translator.translate_all()
    elapsed_time = time.time() - start_time
    logger.info(f"翻译完成，耗时 {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    asyncio.run(main())