# api/ 目录

## 功能描述
提供 FastAPI HTTP 接口，让前端或其他客户端能通过 HTTP 调用 Agent 功能。

## 文件列表

| 文件名 | 功能 | 主要接口 |
|--------|------|----------|
| `__init__.py` | 应用初始化 | FastAPI app 实例 |
| `main.py` | 服务入口 | `uvicorn.run(app)` |
| `routes/dialogue.py` | 对话接口 | `POST /chat`, `GET /session/{id}/history` |
| `routes/sql.py` | SQL 相关接口（可选） | `POST /execute`, `GET /schema` |
| `routes/admin.py` | 管理接口（可选） | `GET /health`, `DELETE /session/{id}` |
| `schemas/request.py` | 请求模型 | `ChatRequest`, `ExecuteRequest` |
| `schemas/response.py` | 响应模型 | `ChatResponse`, `HistoryResponse` |

## 使用方法

### 启动服务
```bash
# 进入项目根目录
cd E:\LLM_code_general\sqlcode-master

# 启动 FastAPI 服务（开发模式，自动重载）
python api/main.py

# 或使用 uvicorn 直接启动
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 调用接口

#### 1. 发送对话消息
```bash
curl <http://localhost:8000/api/v1/chat> \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user001",
    "message": "列出所有学生",
    "db_id": "student_db"
  }'
```

**响应示例：**
```json
{
  "session_id": "user001",
  "reply": "找到 50 名学生",
  "sql": "SELECT * FROM student;",
  "result": {
    "columns": ["id", "name", "age"],
    "rows": [[1, "张三", 20], ...],
    "row_count": 50
  },
  "success": true,
  "error": null,
  "trace": [...]
}
```

#### 2. 获取会话历史
```bash
curl <http://localhost:8000/api/v1/session/user001/history>
```

#### 3. 健康检查
```bash
curl <http://localhost:8000/api/v1/health>
```

## 会话管理

- 首次调用 `/chat` 时不提供 `session_id`，服务端会自动生成并返回
- 后续调用使用相同的 `session_id` 来维持多轮对话
- 会话默认保存在内存中，重启服务后丢失（后续可接入 Redis 持久化）

## 运行依赖
- fastapi >= 0.115.0
- uvicorn[standard] >= 0.30.0
- pydantic >= 2.9.0

## 注意事项
- 默认监听 `0.0.0.0:8000`（可在 `main.py` 或 `config/api_config.yaml` 中修改）
- 开发模式下启用 `--reload`，代码修改后自动重启
- 生产环境建议关闭 `--reload`，并使用多进程（如 `uvicorn --workers 4`）
