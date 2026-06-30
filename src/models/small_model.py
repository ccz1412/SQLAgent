"""
小模型（Llama-3.1-8B + LoRA）本地加载与推理模块

支持功能：
1. 加载 Llama-3.1-8B 基础模型（dep/model/）
2. 加载 LoRA 权重（exp/outputs/sql2sr_lora/）
3. 4-bit/8-bit 量化（节省显存）
4. 统一的 generate() 接口
5. ReAct 推理模式（思考链）

使用示例：
    from src.models.small_model import SmallModel
    
    # 加载模型（自动选择最优设备）
    model = SmallModel(
        base_model_path="dep/model/Meta-Llama-3___1-8B-Instruct",
        lora_path="exp/outputs/sql2sr_lora",  # 可选
        device="auto",        # "cuda:0", "cpu", "auto"
        load_in_4bit=True   # 4-bit 量化，节省显存
    )
    
    # 生成文本
    response = model.generate(
        prompt="列出所有计算机科学专业的学生",
        max_tokens=512,
        temperature=0.1
    )
    print(response)
"""

import torch
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Any

warnings.filterwarnings("ignore")


class SmallModel:
    """
    小模型封装类（Llama-3.1-8B + LoRA）
    
    职责：
    1. 加载本地模型（支持量化）
    2. 加载 LoRA 权重（如果提供）
    3. 提供 generate() 接口
    4. 管理推理上下文
    """
    
    def __init__(
        self,
        base_model_path: str,
        lora_path: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        torch_dtype: str = "bfloat16"
    ):
        """
        初始化小模型
        
        参数：
            base_model_path: 基础模型路径（dep/model/Meta-Llama-3___1-8B-Instruct/）
            lora_path: LoRA 权重路径（exp/outputs/sql2sr_lora/），可选
            device: 设备（"cuda:0", "cpu", "auto"）
            load_in_4bit: 是否使用 4-bit 量化（推荐，显存 ~4-5GB）
            load_in_8bit: 是否使用 8-bit 量化（显存 ~8GB）
            torch_dtype: 模型精度（"bfloat16", "float16", "float32"）
        """
        self.base_model_path = Path(base_model_path)
        self.lora_path = Path(lora_path) if lora_path else None
        self.device = self._get_device(device)
        
        # 检查路径
        if not self.base_model_path.exists():
            raise FileNotFoundError(f"基础模型不存在: {self.base_model_path}")
        
        if self.lora_path and not self.lora_path.exists():
            raise FileNotFoundError(f"LoRA 权重不存在: {self.lora_path}")
        
        # 加载模型和分词器
        self.tokenizer, self.model = self._load_model(
            load_in_4bit, load_in_8bit, torch_dtype
        )
        
        print(f"[SmallModel] 模型加载完成 | 设备: {self.device} | LoRA: {lora_path or '无'}")
    
    def _get_device(self, device: str) -> torch.device:
        """自动选择设备"""
        if device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self, load_in_4bit: bool, load_in_8bit: bool, torch_dtype: str):
        """
        加载模型和分词器
        
        步骤：
        1. 检查依赖（transformers, peft, bitsandbytes）
        2. 加载分词器
        3. 加载基础模型（支持量化）
        4. 加载 LoRA 权重（如果提供）
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel, PeftConfig
        except ImportError as e:
            raise ImportError(
                f"缺少依赖：{e}\n"
                f"请运行：pip install transformers peft accelerate bitsandbytes"
            )
        
        print(f"[SmallModel] 正在加载分词器：{self.base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.base_model_path),
            trust_remote_code=True
        )
        
        # 设置 padding 侧（Llama 使用 left padding）
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        print(f"[SmallModel] 正在加载基础模型：{self.base_model_path}")
        print(f"          量化设置：4-bit={load_in_4bit}, 8-bit={load_in_8bit}")
        
        # 构建加载参数
        load_kwargs = {
            "device_map": "auto" if self.device.type == "cuda" else None,
            "trust_remote_code": True
        }
        
        # 量化配置
        if load_in_4bit and self.device.type == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print(f"[SmallModel] 使用 4-bit 量化（BitsAndBytes）")
            except ImportError:
                print(f"[SmallModel] 警告：bitsandbytes 未安装，禁用 4-bit 量化")
                load_in_4bit = False
        
        if load_in_8bit and not load_in_4bit and self.device.type == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                print(f"[SmallModel] 使用 8-bit 量化")
            except ImportError:
                print(f"[SmallModel] 警告：bitsandbytes 未安装，禁用 8-bit 量化")
                load_in_8bit = False
        
        # 如果未量化，设置 dtype
        if not load_in_4bit and not load_in_8bit:
            load_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
            print(f"[SmallModel] 使用完整精度：{torch_dtype}")
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            str(self.base_model_path),
            **load_kwargs
        )
        
        # 加载 LoRA 权重
        if self.lora_path and self.lora_path.exists():
            print(f"[SmallModel] 正在加载 LoRA 权重：{self.lora_path}")
            try:
                model = PeftModel.from_pretrained(model, str(self.lora_path))
                print(f"[SmallModel] LoRA 权重加载成功")
            except Exception as e:
                print(f"[SmallModel] 警告：LoRA 加载失败：{e}")
                print(f"          将使用基础模型继续...")
        
        # 设置为评估模式
        model.eval()
        
        # 如果未使用 device_map="auto"，手动移动到设备
        if load_kwargs.get("device_map") is None:
            model = model.to(self.device)
            print(f"[SmallModel] 模型已移动到设备：{self.device}")
        
        return tokenizer, model
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True,
        stop_tokens: Optional[List[str]] = None
    ) -> str:
        """
        生成文本
        
        参数：
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 温度（越低越确定，越高越随机）
            top_p: 核采样参数
            do_sample: 是否使用采样（False 则贪心解码）
            stop_tokens: 停止 token 列表（可选）
        
        返回：
            生成的文本（str）
        """
        # 构造输入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # 移除 None 值
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # 解码（只取新生成的 token）
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 去除停止 token
        if stop_tokens:
            for stop in stop_tokens:
                if stop in response:
                    response = response[:response.index(stop)]
        
        return response.strip()
    
    def generate_with_chat_template(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> str:
        """
        使用聊天模板生成（推荐用于 Llama 3.1 Instruct）
        
        参数：
            messages: 消息列表，格式：[{
  
            max_tokens: 最大生成 token 数
            temperature: 温度
        
        返回：
            生成的回复（str）
        """
        # 应用聊天模板
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 生成
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response
    
    def react_think(self, observation: str, thought_prompt: str) -> Dict[str, str]:
        """
        ReAct 推理：Thought 阶段（小模型生成思考）
        
        参数：
            observation: 观察（SQL 执行结果或错误信息）
            thought_prompt: 思考提示（引导小模型生成 Thought）
        
        返回：
            {
                "thought": "思考内容",
                "action": "next_action"  # 可选
            }
        """
        # 构造 ReAct 提示
        prompt = f"""{thought_prompt}

Observation: {observation}

Thought: """
        
        response = self.generate(
            prompt=prompt,
            max_tokens=256,
            temperature=0.1,
            do_sample=False  # 思考阶段使用贪心解码
        )
        
        # 解析 Thought 和 Action
        result = {"thought": response}
        
        # 尝试提取 Action（如果有）
        if "Action:" in response:
            parts = response.split("Action:")
            result["thought"] = parts[0].strip()
            result["action"] = parts[1].strip()
        
        return result
    
    def __del__(self):
        """清理显存"""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
