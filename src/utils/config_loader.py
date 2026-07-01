import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径（相对于项目根目录）。
                     如果为 None，则自动加载 config/ 目录下所有 .yaml 文件并合并。

    Returns:
        配置字典
    """
    if config_path is None:
        # 加载 config/ 下所有 yaml 文件并合并
        config_dir = PROJECT_ROOT / "config"
        if not config_dir.exists():
            return {}

        merged_config: Dict[str, Any] = {}
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    merged_config.update(cfg)
            except Exception as e:
                print(f"[config_loader] 加载配置失败 {yaml_file}: {e}")
        return merged_config

    path = PROJECT_ROOT / config_path
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config if isinstance(config, dict) else {}
    except Exception as e:
        print(f"[config_loader] 加载配置失败 {path}: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
    """
    保存配置字典到 YAML 文件

    Args:
        config: 配置字典
        config_path: 配置文件路径（相对于项目根目录）

    Returns:
        是否保存成功
    """
    path = PROJECT_ROOT / config_path
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        print(f"[config_loader] 保存配置失败 {path}: {e}")
        return False


def load_db_root(
    config_path: str = "config/db_config.yaml",
    dataset: str = "spider"
) -> str:
    """
    从 db_config.yaml 读取数据库根目录

    Args:
        config_path: 配置文件相对路径（相对于项目根目录）
        dataset: 数据集名称（spider / bird）

    Returns:
        数据库根目录相对路径
    """
    config_file = PROJECT_ROOT / config_path
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get(dataset, {}).get("db_root", f"data/{dataset}_databases")
        except Exception as e:
            print(f"[config_loader] 加载数据库配置失败: {e}，使用默认路径")
    return f"data/{dataset}_databases"
