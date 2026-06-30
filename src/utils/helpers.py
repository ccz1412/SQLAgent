"""
辅助函数模块
提供通用的工具函数
"""

import re
from typing import List, Dict, Optional, Tuple


def format_prompt(template: str, **kwargs) -> str:
    """
    格式化 Prompt 模板
    
    Args:
        template: 模板字符串（使用 {variable} 格式）
        **kwargs: 变量值
        
    Returns:
        格式化后的字符串
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"模板变量未提供: {e}")


def clean_sql(sql: str) -> str:
    """
    清理 SQL 语句（去除多余空格、注释等）
    
    Args:
        sql: 原始 SQL
        
    Returns:
        清理后的 SQL
    """
    # 去除首尾空格
    sql = sql.strip()
    
    # 去除 SQL 中的单行注释（-- 注释）
    sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    
    # 去除多余空白（保留一个空格）
    sql = re.sub(r"\s+", " ", sql)
    
    return sql.strip()


def parse_sql_clauses(sql: str) -> Dict[str, str]:
    """
    解析 SQL 语句的各个子句
    
    Args:
        sql: SQL 语句
        
    Returns:
        包含各子句的字典（SELECT, FROM, WHERE, GROUP BY, ORDER BY, LIMIT）
    """
    clauses = {
        "SELECT": "",
        "FROM": "",
        "WHERE": "",
        "GROUP BY": "",
        "ORDER BY": "",
        "LIMIT": ""
    }
    
    # 简单解析（不处理嵌套子查询）
    sql_upper = sql.upper()
    
    # 找到各个关键字的位置
    positions = {}
    for key in clauses.keys():
        pos = sql_upper.find(key)
        if pos != -1:
            positions[key] = pos
    
    # 按位置排序
    sorted_positions = sorted(positions.items(), key=lambda x: x[1])
    
    for i, (key, start_pos) in enumerate(sorted_positions):
        # 找到当前子句的结束位置（下一个关键字的位置）
        if i < len(sorted_positions) - 1:
            end_pos = sorted_positions[i + 1][1]
            clause_content = sql[start_pos + len(key):end_pos].strip()
        else:
            clause_content = sql[start_pos + len(key):].strip()
        
        clauses[key] = clause_content
    
    return clauses


def truncate_text(text: str, max_length: int = 2048, suffix: str = "...") -> str:
    """
    截断文本（保留 suffix）
    
    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后添加的后缀
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_sql_from_response(response: str) -> Optional[str]:
    """
    从模型响应中提取 SQL 语句
    
    支持以下格式：
    - 直接是 SQL
    - 被 ```sql ... ``` 包围
    - 包含 "SELECT" 或 "select" 的段落
    
    Args:
        response: 模型响应文本
        
    Returns:
        提取的 SQL，如果没有找到则返回 None
    """
    # 尝试提取 ```sql ... ``` 包围的内容
    pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return clean_sql(match.group(1))
    
    # 尝试提取 ``` ... ``` 包围的内容（无语言标记）
    pattern = r"```\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        sql = clean_sql(match.group(1))
        if sql.upper().startswith("SELECT") or sql.upper().startswith("WITH"):
            return sql
    
    # 尝试直接查找 SELECT 语句
    lines = response.split("\n")
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
        return clean_sql("\n".join(sql_lines))
    
    return None


# 导出
__all__ = [
    "format_prompt",
    "clean_sql",
    "parse_sql_clauses",
    "truncate_text",
    "extract_sql_from_response"
]
