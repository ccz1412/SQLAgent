"""
意图识别模块 (Intent Detection / Function Calling) v2

功能：
- 识别用户输入的意图：查询数据库 vs 聊天 vs 拒绝
- 非查询请求：明确告知用户不是数据库查询，请重新输入
- 查询请求：转发给 SQL 生成 LLM 处理
- 聊天限定范围：介绍自己、介绍数据库、DB 相关知识、已查表总结
- 非相关话题：委婉拒绝

所有 Prompt 模板统一从 src/prompts/prompt_templates.py 导入。

使用方式：
    from src.agent.intent_detector import IntentDetector, IntentType

    detector = IntentDetector()
    result = detector.detect("列出所有学生")
    if result.intent == IntentType.SQL_QUERY:
        sql = result.sql_query
    else:
        chat_reply = result.reply
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
import json

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入集中式 Prompt
from src.prompts.prompt_templates import (
    INTENT_SYSTEM_PROMPT,
    build_intent_user_prompt,
    REJECT_MESSAGE_TEMPLATE,
    NON_QUERY_REJECT_MESSAGE,
)

# 导入 API 客户端
try:
    from src.call_api.qwen_client import chat as api_chat
except Exception as e:
    api_chat = None


class IntentType(Enum):
    """用户意图类型"""
    SQL_QUERY = "sql_query"       # 数据库查询（走 ReAct Agent）
    CHAT = "chat"                 # 聊天（在允许范围内回答）
    REJECT = "reject"             # 拒绝（超出范围）


class IntentResult:
    """意图识别结果"""
    def __init__(
        self,
        intent: IntentType,
        reply: Optional[str] = None,
        sql_query: Optional[str] = None,
        raw_response: Optional[str] = None,
        confidence: float = 0.0
    ):
        self.intent = intent
        self.reply = reply          # 聊天回复或拒绝消息
        self.sql_query = sql_query  # 提取出的查询意图（可选）
        self.raw_response = raw_response
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "reply": self.reply,
            "sql_query": self.sql_query,
            "confidence": self.confidence
        }


class IntentDetector:
    """
    意图识别器 v2

    使用大模型 API 判断用户输入的意图。
    非查询类输入会明确告知用户不是数据库查询，请重新输入。
    """

    def __init__(self):
        """初始化意图识别器"""
        if api_chat is None:
            raise RuntimeError("API 未正确配置，无法使用意图识别")

        self._schema_info: Optional[str] = None  # 当前连接的数据库 schema 信息
        self._query_history: List[Dict] = []      # 已查询的历史记录

    def set_context(self, schema: Optional[str] = None, history: Optional[List[Dict]] = None):
        """
        设置上下文信息（用于聊天时参考）

        Args:
            schema: 当前数据库的 CREATE TABLE schema
            history: 已完成的查询历史
        """
        self._schema_info = schema
        self._query_history = history or []

    def _build_context_info(self) -> str:
        """构建上下文信息文本"""
        parts = []
        if self._schema_info:
            parts.append(f"当前数据库 Schema:\n{self._schema_info}")
        if self._query_history:
            lines = ["最近的查询历史:"]
            for i, h in enumerate(self._query_history[-5:], 1):
                lines.append(
                    f"  {i}. Q:{h.get('question','')} | "
                    f"SQL:{h.get('sql','')} | 行数:{h.get('row_count','')}"
                )
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def detect(self, user_input: str) -> IntentResult:
        """
        检测用户意图

        Args:
            user_input: 用户输入文本

        Returns:
            IntentResult 对象
        """
        # 构造 Prompt（使用集中式模板）
        context_info = self._build_context_info()
        prompt = build_intent_user_prompt(
            user_input=user_input,
            context_info=context_info
        )
        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        try:
            content, _info = api_chat(messages, temperature=0.1, max_tokens=256)
            result = self._parse_response(content)

            if result is None:
                result = self._fallback_detect(user_input)

            # 对于非查询意图，确保拒绝消息明确告知用户不是数据库查询
            if result.intent == IntentType.REJECT:
                if not result.reply or len(result.reply) < 10:
                    result.reply = NON_QUERY_REJECT_MESSAGE

            return result

        except Exception as e:
            return self._fallback_detect(user_input)

    def _parse_response(self, content: str) -> Optional[IntentResult]:
        """解析模型返回的 JSON"""
        text = content.strip()

        # 去掉可能的 markdown code block
        if "```" in text:
            import re
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        try:
            data = json.loads(text)
            intent_str = data.get("intent", "reject")
            reply = data.get("reply", "")
            confidence = float(data.get("confidence", 0.5))

            intent_map = {
                "sql_query": IntentType.SQL_QUERY,
                "chat": IntentType.CHAT,
                "reject": IntentType.REJECT
            }
            intent = intent_map.get(intent_str, IntentType.REJECT)

            return IntentResult(
                intent=intent,
                reply=reply,
                raw_response=content,
                confidence=confidence
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    def _fallback_detect(self, user_input: str) -> IntentResult:
        """
        规则回退方案（当模型调用失败时）
        """
        input_lower = user_input.lower()

        # 明确的查询意图关键词
        sql_keywords = [
            "查询", "列出", "找", "搜索", "统计", "计算", "多少",
            "select", "find", "show", "list", "count", "average",
            "最大", "最小", "排序", "分组", "汇总",
            "前", "后", "所有", "哪些", "几个", "谁", "哪些人",
            "table", "表", "字段", "列", "数据", "记录", "行"
        ]

        for kw in sql_keywords:
            if kw in user_input:
                return IntentResult(
                    intent=IntentType.SQL_QUERY,
                    reply="好的，我来帮您查询数据库。",
                    confidence=0.7
                )

        # 自我介绍类
        intro_keywords = ["你是谁", "你能做什么", "介绍你自己", "你的能力", "help"]
        for kw in intro_keywords:
            if kw in input_lower:
                return IntentResult(
                    intent=IntentType.CHAT,
                    reply=self._build_intro_reply(),
                    confidence=0.9
                )

        # 数据库知识类
        db_knowledge_keywords = ["sql", "数据库", "join", "索引", "查询优化", "什么是"]
        for kw in db_knowledge_keywords:
            if kw in input_lower:
                return IntentResult(
                    intent=IntentType.CHAT,
                    reply="这是一个很好的数据库相关问题！不过我目前主要专注于帮您执行实际的数据库查询。如果您有具体的数据查询需求，可以直接告诉我。",
                    confidence=0.6
                )

        # 默认拒绝 — 明确告知用户不是数据库查询
        return IntentResult(
            intent=IntentType.REJECT,
            reply=NON_QUERY_REJECT_MESSAGE,
            confidence=0.5
        )

    def _build_intro_reply(self) -> str:
        """构建自我介绍的回复"""
        reply = (
            "我是 Multi-Turn Text-to-SQL 智能助手！\n\n"
            "**我的能力：**\n"
            "- 用自然语言查询数据库（自动生成 SQL 并执行）\n"
            "- 支持多轮对话和追问\n"
            "- 自动分析错误并重新生成 SQL\n\n"
            "**使用方法：**\n"
            '直接告诉我你想查什么，比如："列出所有计算机科学专业的学生"\n\n'
        )
        if self._schema_info:
            reply += "**当前已连接数据库**，可以直接开始查询！"
        else:
            reply += "请先选择要查询的数据库。"

        return reply


__all__ = ["IntentDetector", "IntentType", "IntentResult"]
