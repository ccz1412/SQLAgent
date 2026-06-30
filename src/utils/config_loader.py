"""
配置文件加载模块
支持 YAML 格式，自动合并多个配置文件
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# 项目根目录（相对路径基准）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径（相对于项目根目录）
                     如果为 None，则加载 config/ 下所有 .yaml 文件并合并
    
    Returns:
        配置字典
    """
    if config_path is None:
        # 加载 config/ 下所有 .yaml 文件
        config_dir = _PROJECT_ROOT / "config"
        config = {}
        
        if not config_dir.exists():
            return config
        
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            with open(yaml_file, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # 合并配置（深层合并）
                    _deep_merge(config, file_config)
        
        return config
    else:
        # 加载单个配置文件
        full_path = _PROJECT_ROOT / config_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {full_path}")
        
        with open(full_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        return config if config else {}


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径（相对于项目根目录）
    """
    full_path = _PROJECT_ROOT / config_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def get_project_root() -> Path:
    """获取项目根目录"""
    return _PROJECT_ROOT


def resolve_path(relative_path: str) -> Path:
    """
    将相对路径解析为绝对路径
    
    Args:
        relative_path: 相对于项目根目录的路径
        
    Returns:
        绝对路径 Path 对象
    """
    return _PROJECT_ROOT / relative_path


def _deep_merge(base: Dict, update: Dict) -> None:
    """
    深层合并两个字典（修改 base 就地）
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# 导出
__all__ = ["load_config", "save_config", "get_project_root", "resolve_path"]
