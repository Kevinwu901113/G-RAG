#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查询分类器持续更新调度脚本

该脚本用于周期性地运行持续学习模块，定期更新查询分类器模型。
可以作为独立脚本运行，也可以集成到主系统中。
"""

import os
import sys
import time
import yaml
import argparse
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入持续学习模块
from grag.classifier.continual_trainer import run_continual_learning
from grag.utils.logger_manager import configure_logging, get_logger

# 配置日志
logger = get_logger("continual_update")

def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件时出错: {str(e)}")
        raise

def run_scheduler(config_path, update_interval=3600, max_runs=None, log_dir="logs"):
    """
    运行调度器，周期性地执行持续学习更新
    
    Args:
        config_path: 配置文件路径
        update_interval: 更新间隔（秒），默认为3600秒（1小时）
        max_runs: 最大运行次数，默认为None（无限运行）
        log_dir: 日志目录
    """
    # 配置日志
    configure_logging(log_dir=log_dir)
    
    # 加载配置
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        continual_config = config.get('continual_learning', {})
        
        # 从配置中获取更新间隔（如果存在）
        update_interval = continual_config.get('update_interval_seconds', update_interval)
    else:
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        config = None
    
    logger.info(f"启动持续学习调度器，更新间隔: {update_interval}秒")
    
    run_count = 0
    try:
        while True:
            # 记录开始时间
            start_time = datetime.now()
            logger.info(f"开始第 {run_count + 1} 次持续学习更新，时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 运行持续学习过程
            success = run_continual_learning(config_path=config_path)
            
            # 记录结束时间
            end_time = datetime.now()
            duration = end_time - start_time
            
            if success:
                logger.info(f"第 {run_count + 1} 次持续学习更新成功完成，耗时: {duration}")
            else:
                logger.warning(f"第 {run_count + 1} 次持续学习更新未能完成，耗时: {duration}")
            
            # 增加运行计数
            run_count += 1
            
            # 检查是否达到最大运行次数
            if max_runs is not None and run_count >= max_runs:
                logger.info(f"已达到最大运行次数 {max_runs}，调度器退出")
                break
            
            # 计算下次运行时间
            next_run_time = start_time.timestamp() + update_interval
            current_time = time.time()
            sleep_time = max(0, next_run_time - current_time)
            
            logger.info(f"下次更新将在 {sleep_time:.1f} 秒后进行")
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("收到中断信号，调度器退出")
    except Exception as e:
        logger.error(f"调度器运行时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    """
    主函数，解析命令行参数并运行调度器
    """
    parser = argparse.ArgumentParser(description="查询分类器持续更新调度脚本")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径，默认为config.yaml")
    parser.add_argument("--interval", type=int, help="更新间隔（秒），默认从配置文件读取或使用3600秒")
    parser.add_argument("--max_runs", type=int, help="最大运行次数，默认为无限运行")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录，默认为logs")
    parser.add_argument("--run_once", action="store_true", help="只运行一次，不进行调度")
    
    args = parser.parse_args()
    
    # 配置日志
    configure_logging(log_dir=args.log_dir)
    
    try:
        if args.run_once:
            # 只运行一次
            logger.info("运行单次持续学习更新")
            success = run_continual_learning(config_path=args.config)
            if success:
                logger.info("持续学习更新成功完成")
                return 0
            else:
                logger.warning("持续学习更新未能完成")
                return 1
        else:
            # 运行调度器
            run_scheduler(
                config_path=args.config,
                update_interval=args.interval or 3600,
                max_runs=args.max_runs,
                log_dir=args.log_dir
            )
            return 0
    except Exception as e:
        logger.error(f"运行时出错: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())