"""
SQL 生成模块
调用大模型生成 SQL 语句
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到 sys.path（以便导入 src/call_api）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入 API 客户端
try:
    from src.call_api.qwen_client import chat as api_chat
    from src.call_api.qwen_client import get_client
except Exception as e:
    # 打印实际错误以便排查
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
    生成 SQL 语句
    
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
    
    # 构建 Prompt
    prompt = f"""你是一个 SQL 专家。根据以下数据库 Schema 和用户问题，生成正确的 SQL 查询语句。

数据库 Schema:
{schema}

用户问题:
{question}

要求:
1. 只输出 SQL 语句，不要有任何解释
2. 使用标准 SQL 语法
3. 如果问题不明确，请生成最可能的 SQL

SQL:"""
    
    # 调用 API
    messages = [
        {"role": "system", "content": "你是一个 SQL 专家，只输出 SQL 语句。"},
        {"role": "user", "content": prompt}
    ]
    
    try:
        content, info = api_chat(messages, temperature=temperature, max_tokens=max_tokens)
        
        # 从响应中提取 SQL
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
    纠正错误的 SQL 语句
    
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
        return {
            "sql": None,
            "success": False,
            "error": "API 未正确配置"
        }
    
    # 构建纠错 Prompt
    prompt = f"""你是一个 SQL 纠错专家。以下 SQL 语句有错误，请根据错误信息修正它。

数据库 Schema:
{schema}

用户问题:
{question}

错误的 SQL:
{wrong_sql}

执行错误信息:
{error_message}

要求:
1. 只输出修正后的 SQL 语句
2. 不要有任何解释

修正后的 SQL:"""
    
    messages = [
        {"role": "system", "content": "你是一个 SQL 纠错专家。"},
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
        return {
            "sql": None,
            "success": False,
            "error": str(e)
        }


def _extract_sql(text: str) -> Optional[str]:
    """
    从文本中提取 SQL 语句
    """
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


# 导出
__all__ = ["generate_sql", "correct_sql"]
