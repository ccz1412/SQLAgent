"""
API 请求模型定义
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    """对话请求"""
    session_id: Optional[str] = Field(None, description="会话 ID（首次可为空，服务端生成）")
    message: str = Field(..., description="用户消息")
    db_id: Optional[str] = Field(None, description="数据库 ID（可选，首次指定后后续自动使用）")


class ExecuteRequest(BaseModel):
    """SQL 执行请求"""
    sql: str = Field(..., description="要执行的 SQL")
    db_id: str = Field(..., description="数据库 ID")


class ChatResponse(BaseModel):
    """对话响应"""
    session_id: str = Field(..., description="会话 ID")
    reply: Optional[str] = Field(None, description="自然语言回复")
    sql: Optional[str] = Field(None, description="生成的 SQL（如果有）")
    result: Optional[Dict[str, Any]] = Field(None, description="SQL 执行结果（如果有）")
    success: bool = Field(..., description="是否成功")
    error: Optional[str] = Field(None, description="错误信息（如果有）")
    trace: Optional[List[Dict]] = Field(None, description="推理轨迹")
    correction_trace: Optional[List[Dict]] = Field(None, description="纠错轨迹")
    turns_used: Optional[int] = Field(None, description="使用的推理轮数")


class HistoryResponse(BaseModel):
    """历史记录响应"""
    session_id: str
    turns: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "0.1.0"
    models_loaded: Dict[str, bool] = {}
