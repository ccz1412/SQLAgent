# Multi-Turn Text-to-SQL Agent

基于 ReAct 架构的多轮对话 Text-to-SQL 系统，支持本地小模型（Llama-3.1-8B + LoRA）和远程大模型 API（智谱 AI）。

## 功能特性

- ✅ **多轮对话**：支持追问、指代消解（"上一个"、"它"）
- ✅ **模型选择**：本地小模型（低成本）或 API 大模型（高质量）
- ✅ **ReAct 推理**：生成 → 执行 → 纠错 → 再生成
- ✅ **Clause 级纠错**：定位错误 clause，针对性纠正（预留 GRPO 接口）
- ✅ **长期记忆**：短期记忆（当前对话）+ 长期记忆（ChromaDB，可选）
- ✅ **评估模块**：Exact Match、Execution Accuracy、Clause Accuracy
- ✅ **API 服务**：FastAPI 接口，支持 HTTP 调用

## 项目结构

```
E:\LLM_code_general\sqlcode-master\
├── config\                    # 配置文件
│   ├── model_config.yaml      # 模型配置（路径、设备、量化）
│   ├── api_config.yaml       # API 配置（智谱 AI）
│   ├── agent_config.yaml     # Agent 超参数
│   └── db_config.yaml        # 数据库配置
├── src\
│   ├── models\             # 模型层
│   │   ├── small_model.py   # 本地小模型加载（Llama-3.1-8B + LoRA）
│   │   ├── large_model.py   # API 大模型调用（智谱 AI）
│   │   └── model_loader.py  # 统一模型加载器
│   ├── agent\              # Agent 核心
│   │   ├── sql_generator.py         # V1（API only）
│   │   ├── sql_generator_v2.py     # V2（支持本地/API）
│   │   ├── react_agent.py            # V1（API only）
│   │   └── react_agent_v2.py       # V2（支持本地/API）
│   ├── dialogue\            # 多轮对话
│   │   └── dialogue_manager.py     # 对话状态机、追问检测
│   ├── correction\          # SQL 纠错
│   │   └── clause_corrector.py    # Clause 级纠错（规则 + API + GRPO）
│   ├── memory\             # 记忆系统
│   │   └── memory_manager.py      # 短期/长期记忆、Schema 缓存
│   ├── execution\          # SQL 执行
│   │   └── sql_executor.py        # 安全执行、结果格式化
│   ├── evaluation\         # 评估模块
│   │   ├── metrics.py      # 评估指标
│   │   ├── evaluator.py    # 评估器
│   │   └── case_analyzer.py     # Case 分析器
│   └── utils\              # 工具类
│       ├── logger.py
│       ├── config_loader.py
│       └── helpers.py
├── api\                       # FastAPI 服务
│   ├── main.py
│   ├── routes\dialogue.py
│   └── schemas\
├── dep\                       # 依赖（模型文件）
│   └── model\
│       └── Meta-Llama-3___1-8B-Instruct\  # 基础模型
├── exp\                       # 实验输出
│   └── outputs\
│       └── sql2sr_lora\                      # LoRA 权重
├── run_app.py                # 运行入口
├── requirements.txt          # 依赖包
└── README.md
```

## 安装依赖

### 选项 1：仅 API 模式（推荐，快速）

```bash
cd E:\LLM_code_general\sqlcode-master
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 选项 2：本地小模型模式（需要 GPU）

```bash
cd E:\LLM_code_general\sqlcode-master
python -m venv venv
venv\Scripts\activate

# 安装基础依赖
pip install -r requirements.txt

# 安装本地模型依赖（需要 GPU 和 CUDA）
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate bitsandbytes
```

## 配置

### 1. API 配置（智谱 AI）

编辑 `config/api_config.yaml`：

```yaml
api:
  base_url: "https://open.bigmodel.cn/api/paas/v4"
  model_name: "glm-4-flash"
  api_key: "your-api-key-here"
```

### 2. 模型配置（本地小模型）

编辑 `config/model_config.yaml`：

```yaml
small_model:
  base_model_path: "dep/model/Meta-Llama-3___1-8B-Instruct"
  lora_path: "exp/outputs/sql2sr_lora"
  device: "auto"  # auto, cuda, cpu
  load_in_4bit: true  # 4-bit 量化（显存 ~4-5GB）
```

## 使用指南

### 模式 1：直接运行 Agent（命令行）

```bash
# 使用 API 大模型（默认）
python run_app.py agent --question "列出所有学生" --db_id student_db

# 使用本地小模型
python run_app.py agent --question "列出所有学生" --db_id student_db --use_small_model
```

**输出示例**：

```
=== ReAct Agent 测试 ===
问题: 列出所有学生
数据库: student_db
模型: 远程大模型 (智谱 AI API)

[第 1 轮] 生成 SQL...
  SQL: SELECT * FROM students
[第 1 轮] 执行 SQL...
  执行成功！返回 10 行

=== 结果 ===
成功: True
SQL: SELECT * FROM students
返回行数: 10
执行时间: 0.123s
```

### 模式 2：多轮对话（交互式）

```bash
# 启动多轮对话
python run_app.py dialogue --db_id student_db

# 或使用本地小模型
python run_app.py dialogue --db_id student_db --use_small_model
```

**对话示例**：

```
=== 多轮对话模式 ===
数据库: student_db
模型: 远程大模型 (智谱 AI API)
输入 'quit' 或 'exit' 退出
输入 'reset' 重置对话

用户: 列出所有学生
Agent:
  SQL: SELECT * FROM students
  返回: 10 行

用户: 只看计算机系的  ← 自动识别为追问
Agent:
  SQL: SELECT * FROM students WHERE major = 'CS'
  返回: 3 行

用户: reset  ← 重置对话
对话已重置。

用户: quit
再见！
```

### 模式 3：启动 API 服务

```bash
# 启动 FastAPI 服务
python run_app.py api
```

然后访问 `<http://localhost:8000/docs>` 查看 API 文档。

**API 使用示例**：

```bash
# 发送对话消息
curl -X POST "<http://localhost:8000/api/v1/dialogue/test-session/message>" \
  -H "Content-Type: application/json" \
  -d '{"message": "列出所有学生"}'
```

### 模式 4：运行评估

```bash
# 评估 API 大模型（Spider dev 集）
python run_app.py evaluate --dataset spider --split dev

# 评估本地小模型（限制 10 个样本）
python run_app.py evaluate --dataset spider --split dev --use_small_model --max_samples 10
```

**输出示例**：

```
=== 评估模式 ===
数据集: spider/dev
模型: 远程大模型 (智谱 AI API)

[1/100] 评估样本
  Question: 列出所有学生
  Gold SQL: SELECT * FROM students
  Pred SQL: SELECT * FROM students
  Exact Match: True

...

评估报告
============================================================

数据集：spider/dev
样本数：100

指标：
  Exact Match Rate: 78/100 (78.00%)

错误样本（前 5 个）：
  1. Question: 找出年龄大于 20 的学生
     Gold SQL: SELECT name FROM students WHERE age > 20
     Pred SQL: SELECT * FROM students WHERE age > 20
```

## 核心模块说明

### 1. ReAct Agent（`src/agent/react_agent_v2.py`）

实现 ReAct (Reasoning + Acting) 循环：

```
用户问题
    ↓
生成 SQL（调用大模型或小模型）
    ↓
执行 SQL（安全执行，只允许 SELECT）
    ↓
成功？ → 返回结果
    ↓
失败？ → Clause 级纠错
    ↓
重新生成 SQL
    ↓
达到最大轮数？ → 返回错误
```

### 2. Clause 级纠错（`src/correction/clause_corrector.py`）

定位错误 clause（WHERE, JOIN, GROUP BY, etc.），生成针对性纠正：

- **规则纠错**：基于错误信息（缺少引号、不完整的 WHERE，etc.）
- **API 纠错**：调用大模型生成纠正后的 SQL
- **GRPO 纠错**：预留接口，后续接入训练好的模型

### 3. 长期记忆（`src/memory/memory_manager.py`）

- **短期记忆**：当前对话的所有轮次（存储在内存中）
- **长期记忆**：ChromaDB 向量库（可选，需要安装 `chromadb`）
- **Schema 记忆**：缓存数据库 schema，避免重复查询
- **纠错经验**：存储 (错误SQL, 错误类型, 纠正后SQL) 三元组

### 4. 评估模块（`src/evaluation/`）

- **metrics.py**：Exact Match, Execution Accuracy, Clause Accuracy
- **evaluator.py**：加载数据集、运行评估、生成报告
- **case_analyzer.py**：错误分类、趋势分析

## 测试

### 测试 1：配置加载

```bash
python -c "from src.utils.config_loader import load_config; config = load_config(); print('配置加载成功')"
```

### 测试 2：API 连接

```bash
python -c "from src.call_api.qwen_client import chat; resp, _ = chat([{'role':'user', 'content':'Say OK'}], max_tokens=10); print('API 测试成功:', resp)"
```

### 测试 3：模型加载（本地小模型）

```bash
python -c "from src.models.model_loader import ModelLoader; loader = ModelLoader(); model = loader.get_model(ModelType.SMALL); print('模型加载成功')"
```

⚠️ 需要安装 `torch`, `transformers`, `peft` 等依赖，且需要 GPU。

### 测试 4：端到端（生成 SQL → 执行）

```bash
# 创建测试数据库
python -c "
import sqlite3
conn = sqlite3.connect('test.db')
conn.execute('CREATE TABLE students (id INT, name TEXT, major TEXT)')
conn.execute(\"INSERT INTO students VALUES (1, 'Alice', 'CS')\")
conn.commit()
conn.close()
print('测试数据库创建成功')
"

# 运行测试
python run_app.py agent --question "列出所有学生" --db_id test
```

## 后续工作

### 1. 实现 GRPO 训练集成

在 `src/correction/clause_corrector.py` 中：

```python
def _correct_by_grpo(self, sql: str, error_clause: str, schema: str, question: str) -> str:
    """使用 GRPO 训练好的模型进行纠错"""
    # TODO: 加载 GRPO 模型（LoRA 权重）
    # from src.models.model_loader import get_model, ModelType
    # grpo_model = get_model(ModelType.SMALL, reload=True)
    
    # 构造输入
    input_text = f"错误SQL: {sql}\n错误类型: {error_clause}\n问题: {question}"
    
    # 生成纠正后的 SQL
    corrected_sql = grpo_model.generate(input_text)
    
    return corrected_sql
```

### 2. 添加 ChromaDB 长期记忆

安装依赖：

```bash
pip install chromadb
```

然后在代码中使用：

```python
from src.memory.memory_manager import MemoryManager

mm = MemoryManager(db_id="student_db", use_chroma=True)
mm.add_turn(question="...", sql="...", result=...)
relevant = mm.search_relevant("学生名单")
```

### 3. 完善评估指标

在 `src/evaluation/metrics.py` 中添加：

- **Component-based Accuracy**（Spider 官方指标）
- **Execution Accuracy**（需要连接数据库执行 SQL）
- **Efficiency Metrics**（推理时间、API 调用次数）

## 常见问题

### Q1：API 调用失败？

**A**：检查 `config/api_config.yaml` 中的配置是否正确，特别是 `api_key`。

### Q2：本地小模型加载失败？

**A**：确保已安装 `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`。

### Q3：显存不足？

**A**：在 `config/model_config.yaml` 中设置 `load_in_4bit: true`（4-bit 量化，显存 ~4-5GB）。

### Q4：如何添加新的数据集？

**A**：在 `src/evaluation/evaluator.py` 中的 `_load_dataset()` 方法中添加数据集加载逻辑。

## 联系与贡献

如有问题或建议，请提交 Issue 或 Pull Request。

---

**最后更新**：2026-06-30
