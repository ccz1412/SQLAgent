"""
意图识别模块 (Intent Detection / Function Calling)

功能：
- 识别用户输入的意图：查询数据库 vs 聊天
- 聊天限定范围：介绍自己、介绍数据库、DB 相关知识、已查表总结
- 非相关话题：委婉拒绝

使用方式：
    from src.agent.intent_detector import IntentDetector, IntentType

    detector = IntentDetector()
    result = detector.detect("列出所有学生")
    if result.intent == IntentType.SQL_QUERY:
        # 走 SQL 查询流程
        sql = result.sql_query
    else:
        # 返回聊天回复
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
    意图识别器

    使用大模型 API 判断用户输入的意图，支持 Function Calling 格式输出。
    """

    # 允许聊天的范围定义
    CHAT_SCOPE = """
    你是一个 Text-to-SQL 助手，以下是你**可以回答**的聊天问题：

    1. **关于你自己**
       - "你是谁？" → "我是 Text-to-SQL 智能助手，可以帮你用自然语言查询数据库..."
       - "你能做什么？" → 介绍你的能力：SQL 查询、数据筛选、统计等

    2. **关于当前连接的数据库**
       - "当前连接了什么数据库？"
       - "这个数据库有哪些表？"
       - "xxx 表有哪些列？"

    3. **关于数据库/SQL 知识**
       - "什么是 JOIN？"
       - "SELECT * 和 SELECT id 有什么区别？"
       - "如何优化慢查询？"
       - "什么是索引？"

    4. **关于已查询过的结果总结**
       - "刚才查到的结果有什么特点？"
       - "能帮我总结一下刚才的数据吗？"

    **超出以上范围的问题，请委婉拒绝。**

    拒绝示例：
       - 用户："今天天气怎么样？" → 回答："抱歉，我是数据库查询助手，无法回答天气等与数据库无关的问题。您可以问我关于当前连接数据库的问题，或者让我帮您查询数据。"
       - 用户："推荐一部电影" → 回答："抱歉，我只能回答与数据库、SQL 或当前连接数据相关的知识性问题。如果您想查询数据库中的电影信息，我可以帮助您。"
       - 用户："写一首诗" → 同上委婉拒绝

    **注意**：如果用户的提问看起来像是想要**查询数据库**（例如提到数据、列表、统计、查找等），即使措辞不明确，也应归类为 sql_query。
    """

    SYSTEM_PROMPT = f"""你是一个意图分类器。判断用户的消息属于哪种意图。

{CHAT_SCOPE}

请严格按照以下 JSON 格式输出（不要输出任何其他内容）：
{{"intent": "sql_query|chat|reject", "reply": "...", "confidence": 0.0-1.0}}

判断规则：
- intent 为 "sql_query" 时：reply 可以是简短确认（如"好的，我来帮你查询..."）
- intent 为 "chat" 时：reply 是具体的回答内容
- intent 为 "reject" 时：reply 是礼貌的拒绝消息
- confidence 表示你的判断置信度
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
            history: 已完成的查询历史 [{"question": ..., "sql": ..., "row_count": ...}, ...]
        """
        self._schema_info = schema
        self._query_history = history or []

    def detect(self, user_input: str) -> IntentResult:
        """
        检测用户意图

        Args:
            user_input: 用户输入文本

        Returns:
            IntentResult 对象
        """
        # 构造上下文信息
        context_info = ""
        if self._schema_info:
            context_info += f"\n\n当前数据库 Schema:\n{self._schema_info}"
        if self._query_history:
            context_info += "\n\n最近的查询历史:\n"
            for i, h in enumerate(self._query_history[-5:], 1):  # 最近 5 条
                context_info += f"  {i}. Q:{h.get('question','')} | SQL:{h.get('sql','')} | 行数:{h.get('row_count','')}\n"

        # 构造 Prompt
        prompt = f"""请判断以下用户消息的意图：

{context_info}

用户消息: {user_input}

请直接输出 JSON（不要有任何其他文字）："""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        try:
            content, _info = api_chat(messages, temperature=0.1, max_tokens=256)
            result = self._parse_response(content)

            # 如果解析失败，回退到关键词规则
            if result is None:
                result = self._fallback_detect(user_input)

            return result

        except Exception as e:
            # API 调用失败，回退到规则
            return self._fallback_detect(user_input)

    def _parse_response(self, content: str) -> Optional[IntentResult]:
        """解析模型返回的 JSON"""
        # 尝试提取 JSON
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

            # 映射 intent 字符串到枚举
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

        基于关键词快速判断意图。
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
                    reply="这是一个很好的数据库相关问题！不过我目前主要专注于帮您执行实际的数据库查询。如果您有具体的数据查询需求，可以直接告诉我；如果您想了解某个数据库概念，我也可以简单解答。",
                    confidence=0.6
                )

        # 默认拒绝
        return IntentResult(
            intent=IntentType.REJECT,
            reply="抱歉，我是 Text-to-SQL 智能助手，只能回答与数据库查询、SQL 语言或当前连接数据相关的问题。如果您想查询数据库中的数据，或者有关于 SQL/数据库的知识性问题，我很乐意帮助您！",
            confidence=0.5
        )

    def _build_intro_reply(self) -> str:
        """构建自我介绍的回复"""
        reply = (
            "我是 Multi-Turn Text-to-SQL 智能助手！\n\n"
            "**我的能力：**\n"
            "- 用自然语言查询数据库（自动生成 SQL 并执行）\n"
            "- 支持多轮对话和追问\n"
            "- 自动纠错 SQL 语句\n\n"
            "**使用方法：**\n"
            '直接告诉我你想查什么，比如："列出所有计算机科学专业的学生"\n\n'
        )
        if self._schema_info:
            reply += "**当前已连接数据库**，可以直接开始查询！"
        else:
            reply += "请先选择要查询的数据库。"

        return reply


__all__ = ["IntentDetector", "IntentType", "IntentResult"]
