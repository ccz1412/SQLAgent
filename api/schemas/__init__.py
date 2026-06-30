"""
API 响应模型定义（别名，实际与 request 在同一文件）
"""

from .request import ChatRequest, ExecuteRequest, ChatResponse, HistoryResponse, HealthResponse

__all__ = ["ChatRequest", "ExecuteRequest", "ChatResponse", "HistoryResponse", "HealthResponse"]
