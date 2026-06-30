"""
对话状态机
管理多轮对话的状态转移
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class DialogueState(Enum):
    """对话状态枚举"""
    INIT = "init"                      # 初始/新会话
    UNDERSTANDING = "understanding"    # 解析用户意图
    QUERY_PROCESSING = "query_processing" # 处理 SQL 查询
    CORRECTING = "correcting"          # 正在纠错
    WAITING_RESULT = "waiting_result"  # 等待结果展示
    FOLLOW_UP = "follow_up"            # 处理追问
    FINISHED = "finished"             # 当前查询完成


class DialogueTurn:
    """对话轮次记录"""
    def __init__(
        self,
        turn_id: int,
        user_message: str,
        sql: Optional[str] = None,
        result: Optional[Any] = None,
        is_follow_up: bool = False
    ):
        self.turn_id = turn_id
        self.user_message = user_message
        self.sql = sql
        self.result = result
        self.is_follow_up = is_follow_up
        self.correction_trace: List[Dict] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message,
            "sql": self.sql,
            "result": self.result,
            "is_follow_up": self.is_follow_up,
            "correction_trace": self.correction_trace
        }


class DialogueManager:
    """
    对话管理器
    负责状态转移和对话流程控制
    """
    
    def __init__(self, db_id: str, use_small_model: bool = False, session_id: Optional[str] = None):
        """
        初始化对话管理器
        
        Args:
            db_id: 数据库 ID
            use_small_model: 是否使用本地小模型（True：Llama-3.1-8B，False：智谱 AI API）
            session_id: 会话 ID（用于持久化，可选）
        """
        self.db_id = db_id
        self.use_small_model = use_small_model
        self.session_id = session_id or "default"
        self.state = DialogueState.INIT
        self.turns: List[DialogueTurn] = []
        self.current_turn_id = 0
        self.current_sql: Optional[str] = None
        self.sql_history: List[Dict] = []  # 所有生成的 SQL 及其结果
        
        # 导入 Agent（延迟导入，避免循环导入）
        from src.agent.react_agent_v2 import ReactAgentV2
        self.agent = ReactAgentV2(
            db_id=db_id,
            use_small_model=use_small_model,
            max_iterations=5,
            temperature=0.1
        )
        
        model_type = "本地小模型 (Llama-3.1-8B + LoRA)" if use_small_model else "远程大模型 (智谱 AI API)"
        logger.info(f"DialogueManager 初始化完成 | 数据库: {db_id} | 模型: {model_type}")
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        处理用户消息（主入口）
        
        Args:
            message: 用户消息
            
        Returns:
            包含 sql, result, success, trace 的字典
        """
        # 判断是否为追问
        is_follow_up = self._is_follow_up(message)
        
        if is_follow_up:
            # 追问：需要解析指代
            resolved_message = self._resolve_reference(message)
            logger.info(f"追问检测：'{message}' → '{resolved_message}'")
            
            # 修改上一轮 SQL（简单情况）或生成新 SQL
            if self._is_simple_modification(message):
                # 简单修改：直接修改上一轮 SQL
                new_sql = self._modify_sql(self.current_sql, message)
                return self._execute_and_respond(new_sql, message, is_follow_up=True)
            else:
                # 复杂追问：需要重新生成 SQL
                return self._run_agent(resolved_message, is_follow_up=True)
        else:
            # 新查询：直接运行 Agent
            return self._run_agent(message, is_follow_up=False)
    
    def _is_follow_up(self, message: str) -> bool:
        """
        判断是否为追问（简化版：基于规则）
        
        规则：
        1. 如果当前有 SQL 历史，且消息中包含"它"、"这个"、"那个"等代词
        2. 或者消息很短（<10 个字符）且不是完整的问句
        """
        if not self.sql_history:
            return False
        
        # 简单规则：包含指代代词
        reference_words = ["它", "这个", "那个", "上一个", "前面", "刚才"]
        for word in reference_words:
            if word in message:
                return True
        
        # 简单规则：消息很短
        if len(message) < 10 and self.current_sql:
            return True
        
        return False
    
    def _resolve_reference(self, message: str) -> str:
        """
        解析指代（简化版）
        
        将"它"、"上一个"等指代词替换为具体内容
        """
        if not self.current_sql:
            return message
        
        # 简单替换：将"它"替换为"上一轮查询"
        resolved = message
        if "它" in message or "这个" in message:
            # 在 SQL 前添加注释，帮助模型理解上下文
            resolved = f"[上下文：上一轮 SQL: {self.current_sql}]\n{message}"
        
        return resolved
    
    def _is_simple_modification(self, message: str) -> bool:
        """
        判断是否为简单修改（可以通过字符串替换完成）
        """
        # 简单修改关键词
        modify_words = ["改成", "改为", "只要", "不要", "限制", "增加", "添加"]
        for word in modify_words:
            if word in message:
                return True
        return False
    
    def _modify_sql(self, sql: str, message: str) -> str:
        """
        修改 SQL（简化版：基于规则）
        """
        # 这里应该调用模型来修改，但简化版先返回原 SQL
        # TODO: 接入 Clause 级纠错模块
        return sql
    
    def _run_agent(self, message: str, is_follow_up: bool) -> Dict[str, Any]:
        """
        运行 ReAct Agent
        """
        logger.info(f"运行 Agent，问题: {message}")
        
        # 调用 Agent
        response = self.agent.run(message)
        
        # 记录对话轮次
        self.current_turn_id += 1
        turn = DialogueTurn(
            turn_id=self.current_turn_id,
            user_message=message,
            sql=response.get("sql"),
            result=response.get("result"),
            is_follow_up=is_follow_up
        )
        self.turns.append(turn)
        
        # 更新当前 SQL 和历史
        if response.get("sql"):
            self.current_sql = response["sql"]
            self.sql_history.append({
                "turn_id": self.current_turn_id,
                "sql": response["sql"],
                "result": response.get("result")
            })
        
        return response
    
    def _execute_and_respond(self, sql: str, message: str, is_follow_up: bool) -> Dict[str, Any]:
        """
        执行 SQL 并响应
        """
        from src.execution.sql_executor import SQLExecutor
        
        executor = SQLExecutor.from_db_id(self.db_id)
        result = executor.execute(sql)
        
        response = {
            "success": result.success,
            "sql": sql,
            "result": {
                "rows": result.rows,
                "columns": result.columns,
                "row_count": result.row_count
            } if result.success else None,
            "error": result.error,
            "is_follow_up": is_follow_up
        }
        
        # 记录对话轮次
        self.current_turn_id += 1
        turn = DialogueTurn(
            turn_id=self.current_turn_id,
            user_message=message,
            sql=sql,
            result=response["result"],
            is_follow_up=is_follow_up
        )
        self.turns.append(turn)
        
        if sql:
            self.current_sql = sql
            self.sql_history.append({
                "turn_id": self.current_turn_id,
                "sql": sql,
                "result": response.get("result")
            })
        
        return response
    
    def get_state(self) -> DialogueState:
        """获取当前状态"""
        return self.state
    
    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return [turn.to_dict() for turn in self.turns]
    
    def reset(self):
        """重置对话"""
        self.state = DialogueState.INIT
        self.turns = []
        self.current_turn_id = 0
        self.current_sql = None
        self.sql_history = []


# 导出
__all__ = ["DialogueManager", "DialogueState", "DialogueTurn"]
