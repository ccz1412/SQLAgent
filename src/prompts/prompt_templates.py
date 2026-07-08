"""
集中式 Prompt 模板文件
========================

本文件统一管理项目中所有 LLM/Agent 使用的 Prompt 模板。
所有需要 Prompt 的模块都应从此文件导入，不再在各自文件中硬编码 Prompt。

分类：
    1. 意图检测 (Intent Detection)
    2. SQL 生成 (SQL Generation)
    3. 纠错代理 - 错误分析 (Correction Agent - Error Analysis)
    4. 纠错代理 - 语义检测 (Correction Agent - Semantic Check)
    5. LLM 重新生成 - 基于错误分析 (LLM Regenerate - Error Analysis)
    6. LLM 重新生成 - 基于语义分析 (LLM Regenerate - Semantic Analysis)
    7. Clause 级纠错 (Clause-Level Correction)
    8. 拒绝提示 (Rejection Messages)

使用方式：
    from src.prompts.prompt_templates import (
        build_sql_generation_prompt,
        build_correction_agent_prompt,
        build_correction_regenerate_prompt,
        build_semantic_check_prompt,
        build_semantic_regenerate_prompt,
        ...
    )
"""

# ============================================================================
# 1. 意图检测 (Intent Detection)
# ============================================================================

CHAT_SCOPE_DEFINITION = """
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

INTENT_SYSTEM_PROMPT = f"""你是一个意图分类器。判断用户的消息属于哪种意图。

{CHAT_SCOPE_DEFINITION}

请严格按照以下 JSON 格式输出（不要输出任何其他内容）：
{{"intent": "sql_query|chat|reject", "reply": "...", "confidence": 0.0-1.0}}

判断规则：
- intent 为 "sql_query" 时：reply 可以是简短确认（如"好的，我来帮你查询..."）
- intent 为 "chat" 时：reply 是具体的回答内容
- intent 为 "reject" 时：reply 是礼貌的拒绝消息
- confidence 表示你的判断置信度
"""

INTENT_USER_TEMPLATE = """请判断以下用户消息的意图：

{context_info}

用户消息: {user_input}

请直接输出 JSON（不要有任何其他文字）："""

# 拒绝消息模板
REJECT_MESSAGE_TEMPLATE = (
    "抱歉，我是 Text-to-SQL 智能助手，只能回答与数据库查询、SQL 语言或当前连接数据相关的问题。"
    "如果您想查询数据库中的数据，或者有关于 SQL/数据库的知识性问题，我很乐意帮助您！"
)

NON_QUERY_REJECT_MESSAGE = (
    "抱歉，您的问题不是数据库查询请求。"
    "请描述您想从数据库中查询的具体信息，例如：'列出所有学生'、'统计每个专业的人数'等。"
)

# ============================================================================
# 2. SQL 生成 (SQL Generation) - 给 API 大模型用
# ============================================================================

SQL_GENERATION_SYSTEM_PROMPT = (
    "你是一个专业的 SQL 数据库专家。"
    "你的任务是：根据数据库 Schema 和用户问题，生成一条准确的 SQLite 查询语句。"
    "只输出 SQL 语句，不要包含任何解释、注释或 markdown 格式。"
)

SQL_GENERATION_USER_TEMPLATE = """根据以下数据库 Schema 和用户问题，生成正确的 SQL 查询语句。

【数据库 Schema】
{schema}

【用户问题】
{question}

【要求】
1. 只输出一条 SQL 查询语句，不要有任何解释
2. 使用标准 SQLite 语法
3. 表名和列名必须与 Schema 中完全一致
4. 注意 JOIN 条件、WHERE 过滤、GROUP BY 分组的正确性
5. 如果问题中有数值比较，注意类型匹配

SQL:"""


# ============================================================================
# 3. 纠错代理 - 错误分析 Prompt (Correction Agent - Error Analysis)
#    给本地 Llama 模型用，分析执行失败的原因
# ============================================================================

CORRECTION_AGENT_ANALYSIS_SYSTEM = (
    "你是一个 SQL 纠错分析专家。你的任务是分析 SQL 执行失败的原因，"
    "定位错误位置，并给出诊断报告。"
)

CORRECTION_AGENT_ANALYSIS_TEMPLATE = """[SQL 错误分析任务]

【数据库 Schema】
{schema}

【用户原始问题】
{question}

【错误的 SQL】
{wrong_sql}

【执行错误信息】
{error_message}

请分析上述 SQL 执行失败的原因，输出 JSON 格式：
{{{{
    "error_type": "语法错误|表名错误|列名错误|JOIN错误|类型不匹配|逻辑错误|其他",
    "error_clause": "SELECT|FROM|WHERE|JOIN|GROUP BY|HAVING|ORDER BY|LIMIT|UNKNOWN",
    "error_description": "具体错误原因（一句话中文描述）",
    "fix_suggestion": "修复建议（一句话中文描述）"
}}}}

注意：
- 仔细对照 Schema 中的表名和列名
- 检查 SQL 语法是否正确（引号、括号、关键字等）
- 只输出 JSON，不要有其他内容"""


# ============================================================================
# 4. 纠错代理 - 语义检测 Prompt (Correction Agent - Semantic Check)
#    给本地 Llama 模型用，执行成功后检查语义是否匹配
# ============================================================================

SEMANTIC_CHECK_SYSTEM = (
    "你是一个 SQL 语义审查专家。你的任务是检查生成的 SQL 查询语句"
    "是否在语义上正确回答了用户的自然语言问题。"
)

SEMANTIC_CHECK_TEMPLATE = """[SQL 语义审查任务]

【数据库 Schema】
{schema}

【用户问题】
{question}

【生成的 SQL】
{sql}

【执行结果摘要】
返回了 {row_count} 行数据，列名: {columns}

请审查这条 SQL 的语义是否正确，输出 JSON 格式：
{{{{
    "semantics_correct": true/false,
    "issues": ["问题1", "问题2"],
    "description": "一句话描述审查结论",
    "confidence": 0.0-1.0
}}}}

审查要点：
- WHERE 条件是否与问题中的筛选条件一致
- SELECT 的列是否是问题所要求的
- JOIN 关系是否正确
- 聚合函数（COUNT/SUM/AVG/MAX/MIN）使用是否恰当
- ORDER BY / GROUP BY / LIMIT 是否与问题意图一致
- 如果问题只需要一行结果但返回了多行，是否缺少 LIMIT
- 如果问题需要排序但 SQL 没有 ORDER BY，是否遗漏

注意：只输出 JSON，不要有其他内容"""


# ============================================================================
# 5. LLM 重新生成 - 基于错误分析 (LLM Regenerate - Error Analysis)
#    给 API 大模型用，根据代理的错误分析重新生成 SQL
# ============================================================================

LLM_REGENERATE_ERROR_SYSTEM = (
    "你是一个 SQL 修复专家。根据错误分析报告修正 SQL，使其能正确执行。"
    "只输出修正后的 SQL 语句，不要任何解释。"
)

LLM_REGENERATE_ERROR_TEMPLATE = """请根据以下错误分析，重新生成正确的 SQL 查询语句。

【数据库 Schema】
{schema}

【用户问题】
{question}

【原 SQL（有错误）】
{wrong_sql}

【错误分析报告】
错误类型: {error_type}
错误位置: {error_clause}
错误描述: {error_description}
修复建议: {fix_suggestion}

【要求】
1. 只输出一条修正后的 SQL 查询语句
2. 不要包含任何解释或注释
3. 确保表名和列名与 Schema 完全一致

修正后的 SQL:"""


# ============================================================================
# 6. LLM 重新生成 - 基于语义分析 (LLM Regenerate - Semantic Analysis)
#    给 API 大模型用，根据语义审查结果优化 SQL
# ============================================================================

LLM_REGENERATE_SEMANTIC_SYSTEM = (
    "你是一个 SQL 优化专家。根据语义审查意见优化 SQL 查询，"
    "使其更准确地回答用户问题。只输出优化后的 SQL 语句。"
)

LLM_REGENERATE_SEMANTIC_TEMPLATE = """请根据语义审查意见，优化 SQL 查询语句。

【数据库 Schema】
{schema}

【用户问题】
{question}

【当前 SQL】
{current_sql}

【语义审查意见】
{semantic_issues}

【要求】
1. 只输出一条优化后的 SQL 查询语句
2. 不要包含任何解释或注释
3. 确保语义完全匹配用户问题意图

优化后的 SQL:"""


# ============================================================================
# 7. Clause 级纠错 (Clause-Level Correction)
# ============================================================================

CLAUSE_CORRECTION_SYSTEM = (
    "你是一个 SQL 纠错专家。请根据错误信息和数据库 Schema，修正错误的 SQL。"
)

CLAUSE_CORRECTION_TEMPLATE = """用户问题：{question}

数据库 Schema：
{schema}

错误的 SQL：
{sql}

错误 clause：{error_clause}

错误信息：执行失败

请修正上述 SQL，使其能正确执行并返回用户问题的答案。只输出修正后的 SQL，不要输出其他内容。"""


# ============================================================================
# 8. Prompt 构建函数 (Builder Functions)
# ============================================================================

def build_intent_user_prompt(user_input: str, context_info: str = "") -> str:
    """构建意图检测的用户 Prompt"""
    return INTENT_USER_TEMPLATE.format(
        context_info=context_info,
        user_input=user_input
    )


def build_sql_generation_prompt(schema: str, question: str) -> str:
    """构建 SQL 生成的用户 Prompt"""
    return SQL_GENERATION_USER_TEMPLATE.format(
        schema=schema,
        question=question
    )


def build_correction_agent_prompt(
    schema: str, question: str, wrong_sql: str, error_message: str
) -> str:
    """构建纠错代理的错误分析 Prompt（给本地 Llama 用）"""
    return CORRECTION_AGENT_ANALYSIS_TEMPLATE.format(
        schema=schema,
        question=question,
        wrong_sql=wrong_sql,
        error_message=error_message
    )


def build_correction_regenerate_prompt(
    schema: str, question: str, wrong_sql: str,
    error_type: str, error_clause: str, error_description: str, fix_suggestion: str
) -> str:
    """构建基于错误分析的 LLM 重新生成 Prompt（给 API 大模型用）"""
    return LLM_REGENERATE_ERROR_TEMPLATE.format(
        schema=schema,
        question=question,
        wrong_sql=wrong_sql,
        error_type=error_type,
        error_clause=error_clause,
        error_description=error_description,
        fix_suggestion=fix_suggestion
    )


def build_semantic_check_prompt(
    schema: str, question: str, sql: str,
    row_count: int = 0, columns: str = ""
) -> str:
    """构建语义检测 Prompt（给本地 Llama 用）"""
    return SEMANTIC_CHECK_TEMPLATE.format(
        schema=schema,
        question=question,
        sql=sql,
        row_count=row_count,
        columns=columns
    )


def build_semantic_regenerate_prompt(
    schema: str, question: str, current_sql: str, semantic_issues: str
) -> str:
    """构建基于语义分析的 LLM 重新生成 Prompt（给 API 大模型用）"""
    return LLM_REGENERATE_SEMANTIC_TEMPLATE.format(
        schema=schema,
        question=question,
        current_sql=current_sql,
        semantic_issues=semantic_issues
    )


def build_clause_correction_prompt(
    sql: str, error_clause: str, schema: str, question: str
) -> str:
    """构建 Clause 级纠错 Prompt"""
    return CLAUSE_CORRECTION_TEMPLATE.format(
        question=question,
        schema=schema,
        sql=sql,
        error_clause=error_clause
    )


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 意图检测
    "INTENT_SYSTEM_PROMPT",
    "INTENT_USER_TEMPLATE",
    "CHAT_SCOPE_DEFINITION",
    "REJECT_MESSAGE_TEMPLATE",
    "NON_QUERY_REJECT_MESSAGE",
    "build_intent_user_prompt",
    # SQL 生成
    "SQL_GENERATION_SYSTEM_PROMPT",
    "SQL_GENERATION_USER_TEMPLATE",
    "build_sql_generation_prompt",
    # 纠错代理 - 错误分析
    "CORRECTION_AGENT_ANALYSIS_SYSTEM",
    "CORRECTION_AGENT_ANALYSIS_TEMPLATE",
    "build_correction_agent_prompt",
    # 纠错代理 - 语义检测
    "SEMANTIC_CHECK_SYSTEM",
    "SEMANTIC_CHECK_TEMPLATE",
    "build_semantic_check_prompt",
    # LLM 重新生成 - 错误分析
    "LLM_REGENERATE_ERROR_SYSTEM",
    "LLM_REGENERATE_ERROR_TEMPLATE",
    "build_correction_regenerate_prompt",
    # LLM 重新生成 - 语义分析
    "LLM_REGENERATE_SEMANTIC_SYSTEM",
    "LLM_REGENERATE_SEMANTIC_TEMPLATE",
    "build_semantic_regenerate_prompt",
    # Clause 纠错
    "CLAUSE_CORRECTION_SYSTEM",
    "CLAUSE_CORRECTION_TEMPLATE",
    "build_clause_correction_prompt",
]
