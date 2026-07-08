# Multi-Turn Text-to-SQL Agent (v2)

基于 ReAct 架构的多轮对话 Text-to-SQL 系统，支持意图检测、双模型纠错架构（API 生成 + 本地 Llama 分析审查）、语义检测。

## v2 更新（2026-07-03）

### 核心优化

1. **集中式 Prompt 管理** — 所有 Prompt 模板统一管理在 `src/prompts/prompt_templates.py`，各模块从此文件导入调用，便于维护和调优。

2. **新的纠错流程**
   - 执行失败 → Agent(本地Llama) **分析错误** → LLM(API) **基于诊断重新生成**
   - 执行成功 → Agent(本地Llama) **语义检测** → 如有问题 → LLM(API) **基于语义分析优化**
   - **终止条件 1**：连续执行失败 3 次
   - **终止条件 2**：执行成功后，语义检测触发 LLM 重新生成，连续 2 次 SQL 无变化

3. **意图检测优化**
   - 非查询类输入：明确告知用户"不是数据库查询，请重新输入"
   - 使用集中式 Prompt 构建

4. **参考架构**
   - 错误分析流程参考 MAC-SQL 的执行反馈机制
   - 语义检测流程参考 MAGIC 的自省/反思方式
   - Prompt 组织结构参考 SHARE 的分层模板系统

### 修改的文件
| 文件 | 变更说明 |
|------|----------|
| `src/prompts/prompt_templates.py` | **新建** — 集中式 Prompt 模板文件（8类 Prompt + 构建函数） |
| `src/prompts/__init__.py` | **新建** — 模块包文件 |
| `src/agent/react_agent.py` | **重写** — 新纠错流程、Agent分析/语义检测、错误分析+语义检测结果类 |
| `src/agent/sql_generator.py` | **重构** — 移除内联 Prompt，改用集中式模板 |
| `src/agent/intent_detector.py` | **重构** — 移除内联 Prompt，改用集中式模板；非查询拒绝消息优化 |
| `src/correction/clause_corrector.py` | **重构** — 移除内联 `_build_correction_prompt`，改用集中式模板 |
| `README.md` | **更新** — 记录 v2 变更 |

## 功能特性

### 已实现 ✅
- ✅ **意图检测（Intent Detection）**：自动区分"查询数据库"和"聊天"，非查询明确拒绝
- ✅ **多轮对话**：支持追问、指代消解
- ✅ **双模型纠错架构 v2**：API 生成 SQL + 本地 Llama 分析错误/语义检测
- ✅ **Agent 错误分析**：本地 Llama 分析执行失败原因（错误类型、位置、描述、修复建议）
- ✅ **Agent 语义检测**：本地 Llama 审查 SQL 语义是否匹配用户问题
- ✅ **LLM 基于分析重新生成**：API 大模型根据 Agent 的诊断报告重新生成 SQL
- ✅ **优先纠错终止条件**：连续失败 3 次或语义检测 2 次无变化
- ✅ **Clause 级纠错**：定位错误 clause，针对性纠正
- ✅ **集中式 Prompt 管理**：所有 Prompt 统一在 `src/prompts/prompt_templates.py`
- ✅ **模型选择**：本地小模型（低成本）或 API 大模型（高质量）
- ✅ **长期记忆**：短期记忆（当前对话）+ 长期记忆（ChromaDB，可选）
- ✅ **评估模块**：Exact Match、Execution Accuracy、Clause Accuracy
- ✅ **API 服务**：FastAPI 接口，支持 HTTP 调用
- ✅ **Web 界面**：Streamlit 前端（可选）

### 开发中 🚧
- 🚧 完善 Clause 级纠错（GRPO 训练的小模型集成）
- 🚧 指代消解（规则版已实现，计划接入小模型）
- 🚧 长期记忆系统（ChromaDB 向量库完善）

## 项目架构

```
用户 → [意图检测] → 非查询? → REPLACE提示重新输入
                   → 查询? → [ReAct Agent v2]

ReAct Agent v2 流程：
  API大模型生成SQL → 执行 → 成功?
                             ├── 是 → Agent(本地Llama)语义检测 → 语义OK?
                             │                    ├── 是 → 返回结果
                             │                    └── 否 → LLM(API)基于分析优化 → 重新执行
                             │                              └── 第2次SQL无变化? → 接受结果
                             └── 否 → Agent(本地Llama)错误分析 → LLM(API)基于诊断重新生成 → 重新执行
                                      └── 连续失败3次? → 终止纠错

终止条件：
  - 连续执行失败 3 次
  - 语义检测触发的第 2 次 LLM 重新生成后 SQL 无变化
  - 达到最大推理轮数（默认 10）
```

## Prompt 管理

所有 Prompt 模板集中在 `src/prompts/prompt_templates.py`，按功能分为 8 类：

| 类别 | 说明 | 使用者 |
|------|------|--------|
| 意图检测 Prompt | 判断用户输入是否为数据库查询 | `intent_detector.py` |
| SQL 生成 Prompt | 引导 LLM 生成 SQL | `sql_generator.py`, `react_agent.py` |
| 错误分析 Prompt | Agent 分析执行失败原因 | `react_agent.py` (Agent = 本地 Llama) |
| 语义检测 Prompt | Agent 审查 SQL 语义 | `react_agent.py` (Agent = 本地 Llama) |
| 基于错误重新生成 | LLM 根据诊断修正 SQL | `react_agent.py` (LLM = API 大模型) |
| 基于语义重新生成 | LLM 根据语义审查优化 SQL | `react_agent.py` (LLM = API 大模型) |
| Clause 级纠错 | 细粒度定位和修正 SQL 子句 | `clause_corrector.py` |
| 拒绝消息 | 非查询输入的拒绝模板 | `intent_detector.py`, `prompt_templates.py` |

## 项目结构

```
sqlcode-master/
├── config/                  # 配置文件（YAML 格式）
├── src/
│   ├── prompts/          # 【v2 新增】集中式 Prompt 模板
│   │   └── prompt_templates.py   # 所有 Prompt + 构建函数
│   ├── agent/            # Agent 核心模块（ReAct 循环、意图检测、SQL 生成）
│   ├── correction/       # Clause 级 SQL 纠错模块
│   ├── dialogue/         # 多轮对话管理（状态机、上下文）
│   ├── execution/        # SQL 执行引擎
│   ├── models/           # 模型加载（LargeModel/SmallModel）
│   ├── utils/            # 通用工具（日志、配置加载、辅助函数）
│   ├── train/            # 模型训练模块
│   └── call_api/         # 大模型 API 调用
├── api/                   # FastAPI 服务层
├── dep/model/             # 模型文件
├── data/                  # 数据集和数据库
├── exp/                   # 实验输出
└── run_app.py             # 运行入口
```

## 安装步骤

### 1. 安装依赖
```bash
cd sqlcode-master
pip install -r requirements.txt
```

### 2. 配置模型
编辑 `config/model_config.yaml`：
- 设置小模型路径（`dep/model/Meta-Llama-3___1-8B-Instruct`）
- 设置 LoRA 权重路径（`exp/outputs/sql2sr_lora/`）
- 设置大模型 API 配置（`config/api_config.yaml`）

### 3. 准备数据库
- 将 SQLite 数据库文件放在 `data/database/` 目录下

## 使用方法

### 模式 1：启动 API 服务（推荐）
```bash
python run_app.py api
# 服务地址：http://localhost:8000
# API 文档：http://localhost:8000/docs
```

调用接口示例：
```bash
curl http://localhost:8000/api/v1/chat \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test001","message":"列出所有学生","db_id":"student_db"}'
```

### 模式 2：命令行 Agent
```bash
python run_app.py agent --question "列出所有学生" --db_id student_db
```

### 模式 3：交互式多轮对话
```bash
python run_app.py dialogue --db_id student_db
```

## 核心模块说明

### 1. ReAct Agent v2 (`src/agent/react_agent.py`)
**双模型协作 + 结构化错误分析**：
- **Agent (本地 Llama)**：分析执行错误（错误类型/位置/描述/修复建议）+ 语义检测
- **LLM (API 大模型)**：生成 SQL + 基于 Agent 分析重新生成

**错误分析结果** (`ErrorAnalysisResult`):
```json
{"error_type": "列名错误", "error_clause": "SELECT", "error_description": "...", "fix_suggestion": "..."}
```

**语义检测结果** (`SemanticCheckResult`):
```json
{"semantics_correct": false, "issues": ["缺少 ORDER BY"], "description": "...", "confidence": 0.7}
```

### 2. Prompt 模板系统 (`src/prompts/prompt_templates.py`)
- 8 类 Prompt 模板 + 对应构建函数
- 修改 Prompt 只需编辑此文件
- 各模块通过构建函数按参数动态生成

### 3. 意图检测 (`src/agent/intent_detector.py`)
- 使用集中式 Prompt 进行意图分类
- 非查询输入：明确提示"请输入数据库查询"
- 回退规则：关键词匹配（模型不可用时）

### 4. SQL 执行引擎 (`src/execution/sql_executor.py`)
- 安全执行（仅允许 SELECT）
- 自动添加 LIMIT
- 返回格式化结果

## 注意事项

1. **路径问题**：所有 Python 文件使用相对路径（相对于项目根目录）
2. **模型加载**：小模型需要本地 GPU，大模型使用 API 调用
3. **数据库**：默认使用 SQLite，可配置其他数据库
4. **多轮对话**：目前使用内存存储，重启后丢失
5. **Prompt 修改**：统一在 `src/prompts/prompt_templates.py` 中修改

## 开发计划

### 下一步 🚀
- [ ] 完善 Clause 级纠错模块（接入 GRPO 训练的小模型）
- [ ] 实现长期记忆系统（ChromaDB）
- [ ] 完善指代消解（接入小模型）
- [ ] 添加单元测试
- [ ] 前端界面（React）

## 许可证

[待定]

## 联系方式

[待定]
