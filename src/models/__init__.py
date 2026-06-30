"""
小模型（Llama-3.1-8B + LoRA）本地加载模块

功能：
1. 加载 Llama-3.1-8B 基础模型（从 dep/model/ 目录）
2. 可选加载 LoRA 权重（从 exp/outputs/sql2sr_lora/）
3. 提供统一的 generate() 接口
4. 支持 4-bit/8-bit 量化（节省显存）

使用方法：
    from src.models.small_model import SmallModel
    
    model = SmallModel(
        base_model_path="dep/model/Meta-Llama-3___1-8B-Instruct",
        lora_path="exp/outputs/sql2sr_lora",  # 可选
        device="cuda:0",
        load_in_4bit=True
    )
    
    response = model.generate("列出所有学生", max_tokens=512)
"""
