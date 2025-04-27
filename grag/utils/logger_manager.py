import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

# 默认日志格式
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

class LoggerManager:
    """
    日志管理器，用于统一管理项目中的日志配置
    """
    
    _instance = None
    _initialized = False
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, log_dir: str = "logs", default_level: str = "info"):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志文件保存目录
            default_level: 默认日志级别
        """
        # 避免重复初始化
        if self._initialized:
            return
            
        self.log_dir = log_dir
        self.default_level = LOG_LEVELS.get(default_level.lower(), logging.INFO)
        
        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 记录当前会话的开始时间，用于生成日志文件名
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化根日志器
        self._setup_root_logger()
        
        self._initialized = True
    
    def _setup_root_logger(self):
        """设置根日志器"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_level)
        
        # 清除现有处理器
        root_logger.handlers = []
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.default_level)
        console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        file_path = os.path.join(self.log_dir, f"grag_{self.session_timestamp}.log")
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(self.default_level)
        file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # 记录会话开始信息
        root_logger.info(f"=== 日志会话开始: {self.session_timestamp} ===")
        root_logger.info(f"日志文件路径: {file_path}")
    
    def get_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        获取指定名称的日志器
        
        Args:
            name: 日志器名称
            level: 日志级别，如果为None则使用默认级别
            
        Returns:
            配置好的日志器
        """
        if name in self._loggers:
            return self._loggers[name]
            
        logger = logging.getLogger(name)
        
        # 设置日志级别
        if level is not None:
            logger.setLevel(LOG_LEVELS.get(level.lower(), self.default_level))
        else:
            logger.setLevel(self.default_level)
            
        # 记录日志器创建信息
        logger.info(f"日志器 '{name}' 已创建")
        
        # 缓存日志器
        self._loggers[name] = logger
        
        return logger
    
    def set_level(self, level: str, logger_name: Optional[str] = None):
        """
        设置日志级别
        
        Args:
            level: 日志级别
            logger_name: 日志器名称，如果为None则设置所有日志器
        """
        log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
        
        if logger_name is None:
            # 设置所有日志器的级别
            for logger in self._loggers.values():
                logger.setLevel(log_level)
            # 同时设置根日志器
            logging.getLogger().setLevel(log_level)
        elif logger_name in self._loggers:
            # 设置指定日志器的级别
            self._loggers[logger_name].setLevel(log_level)
    
    def create_session_log_file(self, prefix: str = "session") -> str:
        """
        创建一个新的会话日志文件
        
        Args:
            prefix: 日志文件名前缀
            
        Returns:
            新日志文件的路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.log_dir, f"{prefix}_{timestamp}.log")
        
        # 记录新会话日志文件创建信息
        root_logger = logging.getLogger()
        root_logger.info(f"创建新会话日志文件: {file_path}")
        
        # 创建并添加新的文件处理器
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(self.default_level)
        file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        
        # 添加到根日志器
        root_logger.addHandler(file_handler)
        
        return file_path
    
    def log_system_info(self):
        """记录系统信息"""
        import platform
        
        logger = logging.getLogger()
        logger.info("=== 系统信息 ===")
        logger.info(f"操作系统: {platform.system()} {platform.release()}")
        logger.info(f"Python版本: {platform.python_version()}")
        logger.info(f"处理器: {platform.processor()}")
        logger.info(f"主机名: {platform.node()}")
        logger.info("===============")
    
    def log_config(self, config: Dict[str, Any]):
        """
        记录配置信息
        
        Args:
            config: 配置字典
        """
        logger = logging.getLogger()
        logger.info("=== 配置信息 ===")
        for key, value in config.items():
            logger.info(f"{key}: {value}")
        logger.info("===============")
    
    def log_operation(self, operation: str, details: Dict[str, Any] = None):
        """
        记录操作信息
        
        Args:
            operation: 操作名称
            details: 操作详情
        """
        logger = logging.getLogger()
        logger.info(f"操作: {operation}")
        if details:
            for key, value in details.items():
                logger.info(f"  {key}: {value}")
    
    def log_performance(self, operation: str, start_time: float, end_time: float = None):
        """
        记录性能信息
        
        Args:
            operation: 操作名称
            start_time: 开始时间戳
            end_time: 结束时间戳，如果为None则使用当前时间
        """
        if end_time is None:
            end_time = time.time()
            
        duration = end_time - start_time
        logger = logging.getLogger()
        logger.info(f"性能: {operation} - 耗时 {duration:.4f} 秒")
    
    def log_error(self, error: Exception, context: str = ""):
        """
        记录错误信息
        
        Args:
            error: 异常对象
            context: 错误上下文
        """
        import traceback
        
        logger = logging.getLogger()
        if context:
            logger.error(f"错误 ({context}): {str(error)}")
        else:
            logger.error(f"错误: {str(error)}")
            
        # 记录详细的堆栈跟踪
        logger.error(traceback.format_exc())


# 创建全局日志管理器实例
logger_manager = LoggerManager()

# 便捷函数
def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """获取指定名称的日志器"""
    return logger_manager.get_logger(name, level)

def set_log_level(level: str, logger_name: Optional[str] = None):
    """设置日志级别"""
    logger_manager.set_level(level, logger_name)

def configure_logging(log_dir: str = "logs", default_level: str = "info"):
    """配置日志系统"""
    global logger_manager
    logger_manager = LoggerManager(log_dir, default_level)
    return logger_manager
