# dialogue/ 目录

## 功能描述
多轮对话管理模块，负责处理用户在多轮对话中的连续查询。

## 文件列表

| 文件名 | 功能 | 主要接口 |
|--------|------|----------|
| `__init__.py` | 模块初始化 | - |
| `dialogue_manager.py` | 对话状态管理 | `DialogueManager`, `DialogueState` |
| `context_manager.py` | 上下文维护 | `ContextManager` |
| `turn_parser.py` | 意图解析（新查询 vs 追问） | `TurnParser` |
| `reference_resolver.py` | 指代消解（"它"、"上一个"等） | `ReferenceResolver` |

## 使用方法

### 基本使用
```python
from src.dialogue.dialogue_manager import DialogueManager

# 初始化对话管理器
dm = DialogueManager(db_id="student_db")

# 处理用户消息
response = dm.process_message("列出所有学生")
print(response["sql"])  # 生成的 SQL
print(response["result"])  # 执行结果

# 多轮对话
response = dm.process_message("只看计算机系的")  # 自动识别为追问
print(response["sql"])  # 修改后的 SQL（WHERE dept='CS'）
```

### 对话状态
```python
# 获取当前状态
state = dm.get_state()  # DialogueState enum

# 状态枚举
DialogueState.INIT          # 初始
DialogueState.UNDERSTANDING # 理解用户意图
DialogueState.QUERY_PROCESSING  # 处理 SQL 查询
DialogueState.CORRECTING      # 正在纠错
DialogueState.FOLLOW_UP       # 处理追问
DialogueState.WAITING_RESULT  # 等待结果展示
DialogueState.FINISHED       # 当前查询完成
```

## 核心功能

### 1. 追问检测
自动识别用户消息是：
- **新查询**："列出所有学生"
- **追问**："只看计算机系的"（指代上一轮查询）

### 2. 指代消解
解析多轮对话中的指代：
- "上一个查询的结果" → 指向 sql_history[-1].result
- "把条件改成北京" → 在上一条 SQL 基础上修改 WHERE
- "按月份分组" → 在上一条 SQL 基础上添加 GROUP BY

### 3. 上下文维护
- 短期记忆：当前对话的所有轮次（最近 N 轮）
- SQL 历史：当前会话中生成的所有 SQL 及其结果
- Schema 缓存：已访问过的数据库 Schema

## 与 Agent 集成
```python
# 在 API 接口中调用
from src.agent.react_agent import ReactAgent
from src.dialogue.dialogue_manager import DialogueManager

@app.post("/chat")
def chat(request: ChatRequest):
    # 获取或创建对话管理器
    dm = get_or_create_dm(request.session_id)
    
    # 处理消息（内部会调用 ReactAgent）
    response = dm.process_message(request.message)
    
    return response
```

## 运行依赖
- Python 3.10+
- src/agent/ 模块
- src/execution/ 模块
- src/memory/ 模块（可选）

## 注意事项
- 指代消解目前使用**规则方法**（正则表达式），后续可接入小模型
- 上下文窗口大小可在 `config/agent_config.yaml` 中配置
- 对话历史默认保存在内存中，重启后丢失（后续可接入持久化）
