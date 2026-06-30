# Multi-Turn Text-to-SQL Agent

基于 ReAct 架构的多轮对话 Text-to-SQL AI 应用，支持 Clause 级纠错。

## 项目背景

本项目在单轮 SQL→SR→纠错科研框架基础上，扩展为完整的多轮对话 Text-to-SQL AI 应用：
- 支持多轮对话中处理多个 SQL 查询
- 集成 Clause 级纠错 Agent
- 支持上下文维护、指代消解、SQL 历史管理
- 提供标准 API 接口供后续前端对接

## 功能特性

### 已实现 ✅
- ✅ ReAct 推理循环（小模型决策 + 大模型执行）
- ✅ 多轮对话管理（状态机、上下文维护）
- ✅ SQL 生成与执行
- ✅ 简单纠错（基于规则）
- ✅ FastAPI HTTP 接口
- ✅ 命令行交互模式

### 开发中 🚧
- 🚧 Clause 级纠错（GRPO 训练的小模型）
- 🚧 指代消解（规则版已实现，计划接入小模型）
- 🚧 长期记忆系统（ChromaDB 向量库）
- 🚧 GRPO 训练模块

## 项目结构

```
E:\LLM_code_general\sqlcode-master\
├── config/                  # 配置文件（YAML 格式）
├── src/
│   ├── agent/            # Agent 核心模块（ReAct 循环、SQL 生成）
│   ├── dialogue/         # 多轮对话管理（状态机、上下文）
│   ├── execution/        # SQL 执行引擎
│   ├── utils/           # 通用工具（日志、配置加载、辅助函数）
│   ├── train/           # 模型训练模块（已有）
│   └── call_api/        # 大模型 API 调用（已有）
├── api/                     # FastAPI 服务层
├── prompts/                # Prompt 模板文件
├── dep/model/              # 模型文件
├── dat/                    # 数据集
├── exp/                    # 实验输出
└── run_app.py             # 运行入口
```

## 安装步骤

### 1. 安装依赖
```bash
cd E:\LLM_code_general\sqlcode-master
pip install -r requirements.txt
```

### 2. 配置模型
编辑 `config/model_config.yaml`：
- 设置小模型路径（`dep/model/Meta-Llama-3___1-8B-Instruct`）
- 设置 LoRA 权重路径（`exp/outputs/sql2sr_lora/`）
- 设置大模型 API 配置（`config/api_config.yaml`）

### 3. 准备数据库
- 将 SQLite 数据库文件放在 `dat/` 目录下
- 或使用 BIRD/Spider 数据集（需指定路径）

## 使用方法

### 模式 1：启动 API 服务（推荐）
```bash
# 启动 FastAPI 服务
python run_app.py api

# 服务地址：<http://localhost:8000>
# API 文档：<http://localhost:8000/docs>
```

#### 调用接口示例
```bash
# 发送对话消息
curl <http://localhost:8000/api/v1/chat> \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test001","message":"列出所有学生","db_id":"student_db"}'

# 获取会话历史
curl <http://localhost:8000/api/v1/session/test001/history>

# 健康检查
curl <http://localhost:8000/api/v1/health>
```

### 模式 2：命令行直接运行 Agent
```bash
# 运行单个查询
python run_app.py agent --question "列出所有学生" --db_id student_db
```

### 模式 3：交互式多轮对话
```bash
# 启动交互式对话
python run_app.py dialogue --db_id student_db

# 然后输入问题：
# 用户: 列出所有学生
# Agent: [生成 SQL 并返回结果]
# 用户: 只看计算机系的
# Agent: [自动识别为追问，修改上一轮 SQL]
# 用户: reset  # 重置对话
# 用户: quit   # 退出
```

## 配置说明

### `config/model_config.yaml`
模型加载参数：
- `small_model`：小模型（Llama-3-8B）配置
- `large_model`：大模型（Qwen-Coder-30B）配置

### `config/api_config.yaml`
API 调用参数：
- `api`：Qwen API 端点、密钥、超时
- `concurrency`：并发调用参数

### `config/agent_config.yaml`
Agent 超参数：
- `react`：ReAct 循环参数（最大轮数、停止条件）
- `sql_generation`：SQL 生成参数（温度、Few-shot）

### `config/db_config.yaml`
数据库配置：
- `default_db_type`：默认数据库类型（sqlite）
- `sqlite`：SQLite 配置
- `spider`：Spider 数据集路径

## 核心模块说明

### 1. ReAct Agent（`src/agent/react_agent.py`）
实现 Reasoning + Acting 循环：
- 生成 SQL → 执行验证 → 纠错 → 再执行
- 最大推理轮数可在配置中设置

### 2. 多轮对话管理（`src/dialogue/dialogue_manager.py`）
- 追问检测（基于规则）
- 指代消解（简化版）
- 上下文维护

### 3. SQL 执行引擎（`src/execution/sql_executor.py`）
- 安全执行（仅允许 SELECT）
- 自动添加 LIMIT
- 返回格式化结果

## 注意事项

1. **路径问题**：所有 Python 文件使用相对路径（相对于项目根目录）
2. **模型加载**：小模型需要本地 GPU 加载，大模型使用 API 调用
3. **数据库**：默认使用 SQLite，可配置其他数据库
4. **多轮对话**：目前使用内存存储，重启后丢失（后续接入持久化）

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
