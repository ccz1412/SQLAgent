"""
日志配置模块
提供统一的日志设置接口
"""

import sys
import os
from pathlib import Path

# 尝试导入 loguru，如果没有则使用标准 logging
try:
    from loguru import logger as _loguru_logger
    
    _HAS_LOGURU = True
except ImportError:
    import logging
    
    _HAS_LOGURU = False

def setup_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_file_prefix: str = "sql_agent"
):
    """
    初始化日志配置
    
    Args:
        log_dir: 日志目录（相对于项目根目录）
        log_level: 日志级别（DEBUG, INFO, WARNING, ERROR）
        log_file_prefix: 日志文件前缀
    """
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent.parent
    log_path = project_root / log_dir
    log_path.mkdir(parents=True, exist_ok=True)
    
    if _HAS_LOGURU:
        # 使用 loguru
        _loguru_logger.remove()  # 移除默认 handler
        
        # 输出到控制台
        _loguru_logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # 输出到文件（按大小轮转）
        _loguru_logger.add(
            str(log_path / f"{log_file_prefix}.log"),
            rotation="100 MB",
            retention="30 days",
            level=log_level,
            encoding="utf-8"
        )
    else:
        # 使用标准 logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stderr),
                logging.FileHandler(str(log_path / f"{log_file_prefix}.log"), encoding="utf-8")
            ]
        )


def get_logger(name: str):
    """
    获取 logger 实例
    
    Args:
        name: logger 名称（通常使用 __name__）
    
    Returns:
        logger 实例
    """
    if _HAS_LOGURU:
        return _loguru_logger.bind(name=name)
    else:
        return logging.getLogger(name)


# 导出
__all__ = ["setup_logger", "get_logger"]
