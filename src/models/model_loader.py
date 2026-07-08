"""
模型加载器（统一入口）

功能：
1. 根据配置自动选择加载小模型或大模型
2. 提供统一的 Model 接口
3. 支持模型热切换

使用示例：
    from src.models.model_loader import get_model, ModelType
    
    # 加载小模型（本地 Llama-3.1-8B + LoRA）
    small_model = get_model(ModelType.SMALL)
    
    # 加载大模型（智谱 AI API）
    large_model = get_model(ModelType.LARGE)
    
    # 统一接口调用
    response = small_model.generate("列出所有学生")
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

# 项目根目录（从当前文件向上查找）
PROJECT_ROOT = Path(__file__).parent.parent.parent


class ModelType(Enum):
    """模型类型枚举"""
    SMALL = "small"  # 本地小模型（Llama-3.1-8B + LoRA）
    LARGE = "large"  # 远程大模型（智谱 AI API）


class ModelConfig:
    """模型配置类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        参数：
            config_path: 配置文件路径（默认：config/model_config.yaml）
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "config" / "model_config.yaml"
        
        # 加载配置
        from src.utils.config_loader import load_config
        config = load_config(str(config_path))
        
        self.small_model_config = config.get("small_model", {})
        self.large_model_config = config.get("large_model", {})
        
        # 小模型路径（转换为绝对路径）
        self.small_base_path = self._to_absolute(
            self.small_model_config.get("base_model_path", "dep/model/Meta-Llama-3___1-8B-Instruct")
        )
        self.small_lora_path = self._to_absolute(
            self.small_model_config.get("lora_path", "exp/outputs/sql2sr_lora")
        )
        self.small_device = self.small_model_config.get("device", "auto")
        self.small_load_in_4bit = self.small_model_config.get("load_in_4bit", True)
        
        # HuggingFace 配置（fast-attention 等）
        hf_config = self.small_model_config.get("huggingface", {})
        self.hf_load_in_4bit = hf_config.get("load_in_4bit", True)
        self.hf_load_in_8bit = hf_config.get("load_in_8bit", False)
        self.hf_torch_dtype = hf_config.get("torch_dtype", "bfloat16")
        self.hf_use_flash_attention_2 = hf_config.get("use_flash_attention_2", True)
        
        # 大模型配置
        self.large_model_name = self.large_model_config.get("model_name", "glm-4-flash")
        self.large_base_url = self.large_model_config.get("base_url", "")
        self.large_api_key = self.large_model_config.get("api_key", "")
    
    def _to_absolute(self, path: str) -> str:
        """转换为绝对路径"""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(PROJECT_ROOT / path)


# 全局模型实例（懒加载）
_small_model_instance = None
_large_model_instance = None


class ModelLoader:
    """
    模型加载器（类接口）
    
    使用示例：
        from src.models.model_loader import ModelLoader
        
        loader = ModelLoader()
        small_model = loader.get_model(ModelType.SMALL)
        large_model = loader.get_model(ModelType.LARGE)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化加载器"""
        self.config = ModelConfig(config_path)
    
    def get_model(self, model_type: ModelType, reload: bool = False):
        """
        获取模型实例
        
        参数：
            model_type: 模型类型
            reload: 是否重新加载
        
        返回：
            模型实例
        """
        return get_model(model_type, reload)
    
    def unload_model(self, model_type: ModelType):
        """卸载模型"""
        unload_model(model_type)
    
    def list_models(self) -> Dict[str, Any]:
        """列出可用模型"""
        return list_available_models()


# 全局单例（向后兼容）
_default_loader = None

def get_default_loader() -> ModelLoader:
    """获取默认加载器（单例）"""
    global _default_loader
    if _default_loader is None:
        _default_loader = ModelLoader()
    return _default_loader()


# 以下是为了向后兼容的函数接口
# 实际使用时推荐使用 ModelLoader 类


def get_model(
    model_type: ModelType,
    reload: bool = False
) -> Any:
    """
    获取模型实例（单例模式）
    
    参数：
        model_type: 模型类型（ModelType.SMALL 或 ModelType.LARGE）
        reload: 是否重新加载模型
    
    返回：
        模型实例（SmallModel 或 LargeModel）
    """
    global _small_model_instance, _large_model_instance
    
    if model_type == ModelType.SMALL:
        if _small_model_instance is None or reload:
            print(f"[ModelLoader] 正在加载小模型...")
            from src.models.small_model import SmallModel
            
            config = ModelConfig()
            _small_model_instance = SmallModel(
                base_model_path=config.small_base_path,
                lora_path=config.small_lora_path,
                device=config.small_device,
                load_in_4bit=config.hf_load_in_4bit,
                load_in_8bit=config.hf_load_in_8bit,
                torch_dtype=config.hf_torch_dtype,
                use_flash_attention_2=config.hf_use_flash_attention_2
            )
        return _small_model_instance
    
    elif model_type == ModelType.LARGE:
        if _large_model_instance is None or reload:
            print(f"[ModelLoader] 正在初始化大模型客户端...")
            from src.models.large_model import LargeModel
            
            _large_model_instance = LargeModel()
        return _large_model_instance
    
    else:
        raise ValueError(f"未知的模型类型：{model_type}")


def unload_model(model_type: ModelType):
    """
    卸载模型（释放显存）
    
    参数：
        model_type: 模型类型
    """
    global _small_model_instance, _large_model_instance
    
    if model_type == ModelType.SMALL and _small_model_instance is not None:
        print(f"[ModelLoader] 正在卸载小模型...")
        del _small_model_instance
        _small_model_instance = None
        
        # 清理显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[ModelLoader] 显存已清理")
        except ImportError:
            pass
    
    elif model_type == ModelType.LARGE:
        # 大模型是 API 调用，无需卸载
        print(f"[ModelLoader] 大模型是 API 客户端，无需卸载")


def list_available_models() -> Dict[str, Any]:
    """
    列出可用的模型
    
    返回：
        模型信息字典
    """
    config = ModelConfig()
    
    info = {
        "small_model": {
            "type": "local",
            "base_path": config.small_base_path,
            "lora_path": config.small_lora_path,
            "device": config.small_device,
            "load_in_4bit": config.small_load_in_4bit,
            "loaded": _small_model_instance is not None
        },
        "large_model": {
            "type": "api",
            "model_name": config.large_model_name,
            "base_url": config.large_base_url,
            "loaded": _large_model_instance is not None
        }
    }
    
    return info


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("模型加载器测试")
    print("=" * 60)
    
    # 列出可用模型
    info = list_available_models()
    print(f"\n可用模型：")
    for name, details in info.items():
        print(f"  {name}:")
        for key, value in details.items():
            print(f"    {key}: {value}")
    
    # 测试大模型（API）
    print(f"\n{'-' * 60}")
    print("测试大模型（智谱 AI API）...")
    large_model = get_model(ModelType.LARGE)
    
    response = large_model.generate(
        messages=[{"role": "user", "content": "你好，请回复'API 测试成功'"}],
        max_tokens=50
    )
    print(f"大模型回复：{response}")
    
    print(f"\n{'=' * 60}")
    print("测试完成！")
    print(f"提示：小模型需要 GPU 和依赖包（torch, transformers, peft）")
