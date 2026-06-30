"""
对话接口路由
POST /api/v1/chat - 多轮对话
GET /api/v1/session/{session_id}/history - 获取会话历史
DELETE /api/v1/session/{session_id} - 清除会话
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Optional
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ..schemas.request import ChatRequest, ChatResponse, HistoryResponse
from src.dialogue.dialogue_manager import DialogueManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# 内存会话存储（简单版，生产环境应接入 Redis）
_sessions: Dict[str, DialogueManager] = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    多轮对话接口
    
    接收用户消息，返回 SQL + 执行结果
    """
    logger.info(f"收到对话请求: session_id={request.session_id}, message={request.message}")
    
    # 获取或创建会话
    session_id = request.session_id or _generate_session_id()
    
    if session_id not in _sessions:
        # 新会话：需要 db_id
        if not request.db_id:
            raise HTTPException(status_code=400, detail="新会话需要提供 db_id")
        
        _sessions[session_id] = DialogueManager(
            db_id=request.db_id,
            session_id=session_id
        )
        logger.info(f"创建新会话: {session_id}, db_id={request.db_id}")
    
    dm = _sessions[session_id]
    
    try:
        # 处理消息
        response = dm.process_message(request.message)
        
        return ChatResponse(
            session_id=session_id,
            reply=_generate_reply(response),
            sql=response.get("sql"),
            result=response.get("result"),
            success=response.get("success", False),
            error=response.get("error"),
            trace=response.get("trace"),
            correction_trace=response.get("correction_trace"),
            turns_used=len(dm.turns) if hasattr(dm, "turns") else None
        )
    
    except Exception as e:
        logger.error(f"处理消息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    """获取会话历史"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    dm = _sessions[session_id]
    return HistoryResponse(
        session_id=session_id,
        turns=dm.get_history()
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """清除会话"""
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"删除会话: {session_id}")
    
    return {"status": "ok", "message": f"会话 {session_id} 已删除"}


def _generate_session_id() -> str:
    """生成会话 ID"""
    import uuid
    return str(uuid.uuid4())


def _generate_reply(response: dict) -> Optional[str]:
    """根据执行结果生成自然语言回复"""
    if not response.get("success"):
        return f"抱歉，执行失败：{response.get('error', '未知错误')}"
    
    result = response.get("result")
    if not result:
        return "查询完成，但没有返回结果。"
    
    row_count = result.get("row_count", 0)
    return f"查询成功，共找到 {row_count} 条记录。"
