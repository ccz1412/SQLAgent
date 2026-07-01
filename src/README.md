# src 目录

## 功能描述
`src/` 是项目的核心源代码目录，包含所有主要功能的实现。

## 子目录结构

| 目录 | 功能 | 状态 |
|------|------|------|
| `src/models/` | 模型层（SmallModel、LargeModel、ModelLoader） | ✅ 已实现 |
| `src/agent/` | Agent 核心模块（ReAct 循环、SQL 生成、意图检测） | ✅ 已实现 |
| `src/dialogue/` | 多轮对话管理（状态机、上下文、指代消解） | ✅ 已实现 |
| `src/correction/` | SQL 纠错（Clause 级、规则、API、GRPO） | 🚧 开发中 |
| `src/memory/` | 记忆系统（短期、长期、Schema 缓存） | 🚧 开发中 |
| `src/execution/` | SQL 执行引擎 | ✅ 已实现 |
| `src/evaluation/` | 评估模块（指标、评估器、Case 分析） | 🚧 开发中 |
| `src/utils/` | 通用工具（日志、配置加载、辅助函数） | ✅ 已实现 |

## 新增功能（2026-06-25）

### 1. 意图检测（Intent Detection）
- 文件：`src/agent/intent_detector.py`
- 功能：自动区分用户输入意图（SQL_QUERY / CHAT / REJECT）
- 集成：`src/dialogue/dialogue_manager.py` 已集成

### 2. 双模型纠错架构
- 文件：`src/agent/react_agent.py`（已更新）
- 功能：API 生成 SQL + 本地 Llama-3.1-8B 审查/纠错
- 配置：`config/agent_config.yaml` 中的 `use_correction_model` 参数

### 3. Streamlit Web 界面
- 文件：`frontend/streamlit_app.py`
- 功能：基于 Streamlit 的 Web 聊天界面

## 运行方式

### 运行 Agent（命令行）
```bash
# 单次查询测试
python run_app.py agent --question "列出所有学生" --db_id student_db

# 多轮对话
python run_app.py dialogue --db_id student_db

# 使用本地小模型
python run_app.py agent --question "列出所有学生" --db_id student_db --use_small_model
```

### 启动 API 服务
```bash
python run_app.py api
# 访问 http://localhost:8000/docs
```

### 启动 Web 界面（Streamlit）
```bash
pip install streamlit
streamlit run frontend/streamlit_app.py
```

## 依赖关系
```
src/agent/        → 依赖 src/utils/, src/execution/, src/models/
src/dialogue/      → 依赖 src/utils/, src/agent/, src/memory/
src/correction/    → 依赖 src/utils/, src/execution/
src/execution/     → 依赖 src/utils/
src/memory/        → 依赖 src/utils/
src/evaluation/    → 依赖 src/utils/, src/execution/
api/               → 依赖 src/agent/, src/dialogue/, src/execution/
frontend/          → 依赖 src/dialogue/, src/agent/
```
