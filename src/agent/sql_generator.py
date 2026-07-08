"""
SQL 生成模块 (v2)
调用 API 大模型生成 SQL 语句

所有 Prompt 模板统一从 src/prompts/prompt_templates.py 导入。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入 Prompt 模板
from src.prompts.prompt_templates import (
    SQL_GENERATION_SYSTEM_PROMPT,
    build_sql_generation_prompt,
    LLM_REGENERATE_ERROR_SYSTEM,
    build_correction_regenerate_prompt,
)

# 导入 API 客户端
try:
    from src.call_api.qwen_client import chat as api_chat
    from src.call_api.qwen_client import get_client
except Exception as e:
    import traceback
    print(f"[ERROR] 导入 qwen_client 失败: {e}")
    traceback.print_exc()
    api_chat = None
    get_client = None


def generate_sql(
    question: str,
    schema: str,
    db_id: str,
    temperature: float = 0.0,
    max_tokens: int = 512
) -> Dict[str, Any]:
    """
    生成 SQL 语句（调用 API 大模型）

    Args:
        question: 用户问题（自然语言）
        schema: 数据库 Schema（CREATE TABLE 语句）
        db_id: 数据库 ID
        temperature: 温度参数
        max_tokens: 最大生成 token 数

    Returns:
        包含 sql, success, error 的字典
    """
    if api_chat is None:
        return {
            "sql": None,
            "success": False,
            "error": "API 未正确配置，请在 config/api_config.yaml 中检查配置"
        }

    # 使用集中式 Prompt 构建
    prompt = build_sql_generation_prompt(schema=schema, question=question)
    messages = [
        {"role": "system", "content": SQL_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    try:
        content, info = api_chat(messages, temperature=temperature, max_tokens=max_tokens)
        sql = _extract_sql(content)
        return {
            "sql": sql,
            "success": sql is not None,
            "error": None if sql else "无法从模型响应中提取 SQL",
            "raw_response": content
        }
    except Exception as e:
        return {
            "sql": None,
            "success": False,
            "error": str(e)
        }


def correct_sql(
    wrong_sql: str,
    error_message: str,
    schema: str,
    question: str,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    纠正错误的 SQL 语句（API 降级备用）

    优先使用 react_agent.py 中的 Agent 分析 + LLM 重新生成流程。
    此函数作为独立的降级备用接口。

    Args:
        wrong_sql: 错误的 SQL
        error_message: 执行错误信息
        schema: 数据库 Schema
        question: 用户问题
        temperature: 温度参数

    Returns:
        包含 sql, success, error 的字典
    """
    if api_chat is None:
        return {"sql": None, "success": False, "error": "API 未正确配置"}

    # 使用错误分析重新生成 Prompt
    prompt = build_correction_regenerate_prompt(
        schema=schema,
        question=question,
        wrong_sql=wrong_sql,
        error_type="未知",
        error_clause="UNKNOWN",
        error_description=error_message[:200],
        fix_suggestion="请根据错误信息修正 SQL"
    )
    messages = [
        {"role": "system", "content": LLM_REGENERATE_ERROR_SYSTEM},
        {"role": "user", "content": prompt}
    ]

    try:
        content, info = api_chat(messages, temperature=temperature)
        sql = _extract_sql(content)
        return {
            "sql": sql,
            "success": sql is not None,
            "error": None if sql else "无法从模型响应中提取 SQL",
            "raw_response": content
        }
    except Exception as e:
        return {"sql": None, "success": False, "error": str(e)}


def _extract_sql(text: str) -> Optional[str]:
    """从文本中提取 SQL 语句"""
    import re

    # 尝试提取 ```sql ... ``` 包围的内容
    pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 尝试提取 ``` ... ``` 包围的内容
    pattern = r"```\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        sql = match.group(1).strip()
        if sql.upper().startswith("SELECT") or sql.upper().startswith("WITH"):
            return sql

    # 尝试直接查找 SELECT 语句
    lines = text.split("\n")
    sql_lines = []
    in_sql = False
    for line in lines:
        if re.search(r"\bSELECT\b", line, re.IGNORECASE):
            in_sql = True
        if in_sql:
            sql_lines.append(line)
            if line.strip().endswith(";"):
                break

    if sql_lines:
        return "\n".join(sql_lines).strip()

    # 如果都没找到，返回整个文本（可能本身就是 SQL）
    text = text.strip()
    if text.upper().startswith("SELECT") or text.upper().startswith("WITH"):
        return text

    return None


__all__ = ["generate_sql", "correct_sql"]
