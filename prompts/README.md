# prompts/ 目录

## 功能描述
存放 Agent 使用的所有 Prompt 模板文件。

## 文件列表

| 文件名 | 功能 | 说明 |
|--------|------|------|
| `system_prompt.txt` | Agent 系统提示词 | 定义 Agent 的角色和行为规则 |
| `sql_generation_prompt.txt` | SQL 生成提示词 | 指导模型生成 SQL（待创建） |
| `clause_correction_prompt.txt` | Clause 纠错提示词 | 指导模型进行细粒度纠错（待创建） |
| `intent_detection_prompt.txt` | 意图检测提示词 | 判断用户消息是新查询还是追问（待创建） |
| `few_shot_examples.json` | Few-shot 示例库 | 提供给模型的示例（可选，待创建） |

## 使用方法

### 在代码中加载 Prompt
```python
from pathlib import Path

# 加载系统提示词
system_prompt_path = Path("prompts/system_prompt.txt")
with open(system_prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()

# 在 Agent 中使用
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_message}
]
```

### Prompt 模板变量
部分 Prompt 包含变量占位符（如 `{schema}`, `{question}`），需要格式化：
```python
from src.utils.helpers import format_prompt

prompt = format_prompt(
    template=sql_generation_prompt,
    schema="CREATE TABLE student (...)",
    question="列出所有学生"
)
```

## 修改 Prompt

1. 直接编辑对应的 `.txt` 文件
2. 重启 API 服务（如果正在运行）
3. 或重新加载 Prompt（如果代码中实现了热加载）

## 注意事项
- 所有 Prompt 文件使用 **UTF-8 编码**
- 变量占位符使用 **花括号**（`{variable}`）
- 系统提示词应尽量 **简洁明确**，避免歧义
- SQL 生成提示词应包含 **数据库 Schema**

## 调试建议
- 在 `config/agent_config.yaml` 中设置 `verbose: true` 查看完整 Prompt
- 使用 FastAPI 的 `/docs` 接口测试不同 Prompt 的效果
- 将成功的 Prompt 版本保存到 `few_shot_examples.json`
