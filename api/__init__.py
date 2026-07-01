"""
API 服务模块初始化
提供 FastAPI HTTP 接口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 创建 FastAPI 应用
app = FastAPI(
    title="Multi-Turn Text-to-SQL Agent API",
    description="多轮对话 Text-to-SQL Agent 系统的 HTTP API 接口",
    version="0.1.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 导入路由
from .routes import dialogue

app.include_router(dialogue.router, prefix="/api/v1", tags=["dialogue"])

# 导入 schemas（供路由使用）
from .schemas import request
