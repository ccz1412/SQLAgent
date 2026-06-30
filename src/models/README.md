"""
模型层 README

## 功能
封装所有模型加载和调用逻辑：
- 小模型（Llama-3.1-8B + LoRA）本地加载
- 大模型（智谱 AI API）远程调用
- 统一的生成接口

## 文件说明

| 文件 | 功能 | 运行设备 |
|------|------|----------|
| `small_model.py` | 加载 Llama-8B + LoRA，本地推理 | GPU（推荐）或 CPU |
| `large_model.py` | 调用智谱 AI API（glm-4-flash） | 任意（API 调用） |
| `model_loader.py` | 统一的模型加载入口 | - |
| `prompt_templates.py` | 所有 Prompt 模板 | - |

## 使用方法

### 1. 加载小模型（本地）
```python
from src.models.small_model import SmallModel

# 加载模型（首次运行会自动下载依赖）
model = SmallModel(
    base_model_path="../dep/model/Meta-Llama-3___1-8B-Instruct",
    lora_path="../exp/outputs/sql2sr_lora",  # 可选
    device="cuda:0",  # 或 "cpu"
    load_in_4bit=True  # 4-bit 量化，节省显存
)

# 生成文本
response = model.generate(
    prompt="列出所有计算机科学专业的学生",
    max_tokens=512,
    temperature=0.1
)
print(response)
```

### 2. 调用大模型（API）
```python
from src.models.large_model import LargeModel

model = LargeModel()
response = model.generate(
    messages=[{"role": "user", "content": "生成 SQL"}],
    max_tokens=1024
)
```

### 3. 统一入口
```python
from src.models.model_loader import get_model

# 根据配置自动选择模型
model = get_model(model_type="small")  # 本地小模型
# 或
model = get_model(model_type="large")  # 远程大模型
```

## 显存要求

| 加载方式 | 显存需求 | 推荐配置 |
|----------|----------|------------|
| FP16（完整精度）| ~16GB | A100 / A6000 |
| INT8 量化 | ~8GB | RTX 3090 / 4070 Ti |
| INT4 量化（推荐）| ~4-5GB | RTX 3060 / 4060 |

## 依赖安装

```bash
# 在虚拟环境中安装
pip install torch transformers peft accelerate bitsandbytes

# 如果使用 CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 注意事项

1. **首次加载会很慢**（需要加载 ~16GB 模型权重）
2. **建议使用 GPU**（CPU 推理速度很慢，约 10-30 秒/句）
3. **4-bit 量化**：使用 `bitsandbytes` 库，显存占用降低 75%
4. **LoRA 权重可选**：如果不提供 `lora_path`，则使用原始 Llama 3.1 能力

## 故障排除

### 问题：CUDA out of memory
**解决**：使用 4-bit 量化 `load_in_4bit=True`

### 问题：模块找不到 `transformers`
**解决**：`pip install transformers`

### 问题：模型加载很慢
**正常**：首次加载需要 ~30 秒，后续会快很多
