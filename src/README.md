# src 目录

## 功能描述
`src/` 是项目的核心源代码目录，包含所有主要功能的实现。

## 子目录结构

| 目录 | 功能 | 状态 |
|------|------|------|
| `src/train/` | 模型训练模块（SFT、GRPO） | ✅ 已有 |
| `src/call_api/` | 大模型 API 调用模块 | ✅ 已有 |
| `src/agent/` | 🆕 Agent 核心模块（ReAct 循环、SQL 生成、纠错） | 🚧 开发中 |
| `src/dialogue/` | 🆕 多轮对话管理（状态机、上下文、指代消解） | 🚧 开发中 |
| `src/memory/` | 🆕 记忆系统（短期、长期、Schema 缓存） | 🚧 开发中 |
| `src/execution/` | 🆕 SQL 执行引擎 | 🚧 开发中 |
| `src/utils/` | 🆕 通用工具（日志、配置加载、辅助函数） | 🚧 开发中 |

## 运行方式

### 训练模型（已有）
```bash
# 进入项目根目录
cd E:\LLM_code_general\sqlcode-master

# SFT 训练
bash src/train/train_sql2sr.sh

# 或 Python 直接运行
python src/train/train_sql2sr.py \
  --model_path dep/model/Meta-Llama-3___1-8B-Instruct \
  --lora_output exp/outputs/sql2sr_lora \
  --data_path exp/ft_data/sql2sr_train_data.json
```

### 调用 Qwen API（已有）
```bash
# 测试 API 连接
python src/call_api/call_qwen.py

# 生成 Spider 轨迹数据
python src/call_api/generate_trajectory.py \
  --input dat/spider_trajectory/train.json \
  --output dat/spider_trajectory/train_trajectory.json \
  --batch_size 32
```

### 运行 Agent（开发中）
```bash
# 单次查询测试
python -m src.agent.react_agent --question "列出所有学生" --db_id student_db

# 启动 API 服务
python api/main.py

# 调用 API
curl <http://localhost:8000/api/v1/chat> \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test001","message":"列出所有学生"}'
```

## 路径说明
- 所有 Python 文件使用**相对导入**（`from.src.xxx import yyy`）
- 配置文件使用**相对路径**（`config/model_config.yaml` 而不是绝对路径）
- 从项目根目录运行脚本

## 依赖关系
```
src/agent/        → 依赖 src/utils/, src/execution/, src/memory/
src/dialogue/      → 依赖 src/utils/, src/memory/
src/execution/     → 依赖 src/utils/
src/memory/        → 依赖 src/utils/
api/               → 依赖 src/agent/, src/dialogue/, src/execution/
```
