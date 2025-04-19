#!/usr/bin/env python3
"""
测试GRAG安装是否成功的简单脚本
"""

import os
import sys
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_dependency(module_name):
    """检查依赖是否已安装"""
    try:
        __import__(module_name)
        logger.info(f"✅ {module_name} 已安装")
        return True
    except ImportError:
        logger.error(f"❌ {module_name} 未安装")
        return False

async def test_grag_imports():
    """测试GRAG的导入"""
    try:
        from grag import LightRAG, QueryParam
        from grag.core.base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage
        from grag.core.storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
        from grag.rag.query import query_with_mode, batch_query
        from grag.rag.embedding import EmbeddingManager
        
        logger.info("✅ GRAG核心模块导入成功")
        return True
    except ImportError as e:
        logger.error(f"❌ GRAG导入失败: {e}")
        return False

async def test_optional_dependencies():
    """测试可选依赖"""
    optional_deps = {
        "neo4j": "Neo4j支持",
        "oracledb": "Oracle支持",
        "unstructured": "DOCX文件支持",
        "lmdeploy": "LMDeploy支持",
        "graspologic": "图分析支持"
    }
    
    results = {}
    for module, description in optional_deps.items():
        try:
            __import__(module)
            logger.info(f"✅ {description} ({module}) 已安装")
            results[module] = True
        except ImportError:
            logger.info(f"ℹ️ {description} ({module}) 未安装 - 这是可选的")
            results[module] = False
    
    return results

async def main():
    """主函数"""
    logger.info("开始测试GRAG安装...")
    
    # 检查Python版本
    python_version = sys.version.split()[0]
    logger.info(f"Python版本: {python_version}")
    
    # 检查核心依赖
    core_deps = [
        "numpy", 
        "tiktoken", 
        "networkx", 
        "nano_vectordb", 
        "tenacity", 
        "openai", 
        "aioboto3", 
        "aiohttp", 
        "ollama", 
        "pydantic", 
        "transformers", 
        "torch"
    ]
    
    all_deps_installed = True
    for dep in core_deps:
        if not check_dependency(dep):
            all_deps_installed = False
    
    # 检查GRAG导入
    grag_imports_ok = await test_grag_imports()
    
    # 检查可选依赖
    optional_deps_results = await test_optional_dependencies()
    
    # 总结
    logger.info("\n--- 测试结果摘要 ---")
    if all_deps_installed:
        logger.info("✅ 所有核心依赖已安装")
    else:
        logger.error("❌ 一些核心依赖缺失")
    
    if grag_imports_ok:
        logger.info("✅ GRAG导入测试通过")
    else:
        logger.error("❌ GRAG导入测试失败")
    
    logger.info("\n可选依赖状态:")
    for module, installed in optional_deps_results.items():
        status = "已安装" if installed else "未安装 (可选)"
        logger.info(f"- {module}: {status}")
    
    if all_deps_installed and grag_imports_ok:
        logger.info("\n🎉 GRAG安装测试成功! 您可以开始使用GRAG了。")
    else:
        logger.error("\n❌ GRAG安装测试失败。请检查上述错误并修复它们。")

if __name__ == "__main__":
    asyncio.run(main())
