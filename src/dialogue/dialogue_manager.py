"""
对话状态机
管理多轮对话的状态转移

功能：
1. 意图识别：区分"查数据库" vs "聊天"（通过 IntentDetector）
2. 对话状态管理：新查询 / 追问 / 纠错
3. 指代消解：处理"它"、"上一个"等代词
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class DialogueState(Enum):
    """对话状态枚举"""
    INIT = "init"                      # 初始/新会话
    UNDERSTANDING = "understanding"    # 解析用户意图
    QUERY_PROCESSING = "query_processing"  # 处理 SQL 查询
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
        is_follow_up: bool = False,
        intent_type: str = ""  # 新增：记录本轮意图类型
    ):
        self.turn_id = turn_id
        self.user_message = user_message
        self.sql = sql
        self.result = result
        self.is_follow_up = is_follow_up
        self.intent_type = intent_type
        self.correction_trace: List[Dict] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message,
            "sql": self.sql,
            "result": self.result,
            "is_follow_up": self.is_follow_up,
            "intent_type": self.intent_type,
            "correction_trace": self.correction_trace
        }


class DialogueManager:
    """
    对话管理器

    职责：
    1. 接收用户消息 → 意图识别 → 路由到对应处理流程
    2. SQL 查询 → ReAct Agent（生成+纠错双模型）
    3. 聊天 → 直接回复（在限定范围内）
    4. 追问检测与指代消解
    """

    def __init__(self, db_id: str, use_small_model: bool = False, session_id: Optional[str] = None):
        """
        初始化对话管理器

        Args:
            db_id: 数据库 ID
            use_small_model: 是否使用本地小模型（用于纠错的 Llama-3.1-8B + LoRA）
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

        # 导入 Agent 和执行器（延迟导入，避免循环导入）
        from src.agent.react_agent import ReactAgent
        from src.execution.sql_executor import SQLExecutor
        from src.agent.intent_detector import IntentDetector

        self.agent = ReactAgent(
            db_id=db_id,
            max_iterations=5,
            temperature=0.1
        )

        self.executor = SQLExecutor.from_db_id(db_id)
        self.schema = self.executor.get_schema()

        # 意图识别器，设置上下文
        self.intent_detector = IntentDetector()
        self._update_intent_context()

        model_info = f"生成模型: API(智谱AI)"
        if use_small_model:
            model_info += f" | 纠错模型: 本地Llama-3.1-8B+LoRA"
        logger.info(f"DialogueManager 初始化完成 | 数据库: {db_id} | {model_info}")

    def _update_intent_context(self):
        """更新意图识别器的上下文信息"""
        self.intent_detector.set_context(
            schema=self.schema,
            history=self.sql_history[-10:] if self.sql_history else []
        )

    def process_message(self, message: str) -> Dict[str, Any]:
        """
        处理用户消息（主入口）

        流程：
        1. 意图识别 → 判断是查询还是聊天
        2. 如果是聊天 → 直接返回聊天回复
        3. 如果是查询 → 走 ReAct Agent 流程

        Args:
            message: 用户消息

        Returns:
            包含 intent, response, sql, result 的字典
        """
        logger.info(f"[DialogueManager] 处理消息: {message}")

        # ===== Step 1: 意图识别 =====
        intent_result = self.intent_detector.detect(message)
        intent_type = intent_result.intent.value
        logger.info(f"[DialogueManager] 意图识别结果: {intent_type} (置信度: {intent_result.confidence:.2f})")

        if intent_result.intent.value == "reject":
            # 拒绝：超出范围
            self.current_turn_id += 1
            return {
                "success": True,
                "intent": "reject",
                "response": intent_result.reply or "抱歉，我只能回答与数据库相关的问题。",
                "sql": None,
                "result": None,
                "is_follow_up": False
            }

        elif intent_result.intent.value == "chat":
            # 聊天：在允许范围内回答
            self.current_turn_id += 1
            turn = DialogueTurn(
                turn_id=self.current_turn_id,
                user_message=message,
                intent_type="chat"
            )
            self.turns.append(turn)

            return {
                "success": True,
                "intent": "chat",
                "response": intent_result.reply,
                "sql": None,
                "result": None,
                "is_follow_up": False
            }

        else:
            # SQL 查询：走 Agent 流程
            # 先判断是否为追问
            is_follow_up = self._is_follow_up(message)

            if is_follow_up:
                resolved_message = self._resolve_reference(message)
                logger.info(f"追问检测：'{message}' → '{resolved_message}'")
                agent_response = self._run_agent(resolved_message, is_follow_up=True)
            else:
                agent_response = self._run_agent(message, is_follow_up=False)

            # 记录意图类型
            agent_response["intent"] = "sql_query"

            # 更新意图识别器的上下文
            if agent_response.get("sql"):
                self.sql_history.append({
                    "question": message,
                    "sql": agent_response["sql"],
                    "row_count": agent_response.get("result", {}).get("row_count", 0) if agent_response.get("result") else 0
                })
                self._update_intent_context()

            return agent_response

    def _is_follow_up(self, message: str) -> bool:
        """
        判断是否为追问

        规则：
        1. 如果当前有 SQL 历史，且消息中包含指代词
        2. 或者消息很短（<10 个字符）且有历史记录
        """
        if not self.sql_history:
            return False

        reference_words = ["它", "这个", "那个", "上一个", "前面", "刚才",
                           "也", "再", "只", "不要", "改成", "改为"]
        for word in reference_words:
            if word in message:
                return True

        if len(message) < 12 and self.current_sql:
            return True

        return False

    def _resolve_reference(self, message: str) -> str:
        """解析指代（将"它"、"上一个"等替换为具体内容）"""
        if not self.current_sql:
            return message

        resolved = message
        if "它" in message or "这个" in message or "那个" in message:
            resolved = f"[上下文：上一轮 SQL: {self.current_sql}]\n{message}"

        return resolved

    def _run_agent(self, message: str, is_follow_up: bool) -> Dict[str, Any]:
        """运行 ReAct Agent"""
        logger.info(f"运行 Agent，问题: {message}")

        try:
            response = self.agent.run(message)
        except Exception as e:
            logger.error(f"Agent 执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql": None,
                "result": None,
                "is_follow_up": is_follow_up
            }

        # 记录对话轮次
        self.current_turn_id += 1
        turn = DialogueTurn(
            turn_id=self.current_turn_id,
            user_message=message,
            sql=response.get("sql"),
            result=response.get("result"),
            is_follow_up=is_follow_up,
            intent_type="sql_query"
        )
        self.turns.append(turn)

        # 更新当前 SQL
        if response.get("sql"):
            self.current_sql = response["sql"]

        # 组装返回格式（兼容原有接口）
        return {
            "success": response.get("success", False),
            "response": response.get("error") or "查询完成",
            "sql": response.get("sql"),
            "result": response.get("result"),
            "error": response.get("error"),
            "is_follow_up": is_follow_up,
            "trace": response.get("trace", []),
            "correction_history": response.get("correction_history", [])
        }

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
        self._update_intent_context()


# 导出
__all__ = ["DialogueManager", "DialogueState", "DialogueTurn"]
