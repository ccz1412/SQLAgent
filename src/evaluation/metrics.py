"""
评估指标模块

包含：
- Exact Match (EM): SQL 字符串完全匹配
- Execution Accuracy (EX): 执行结果匹配
- Clause-level Accuracy: Clause 级准确率
- Correction Success Rate: 纠错成功率
- Average Turns: 平均对话轮数

使用示例：
    from src.evaluation.metrics import exact_match, execution_accuracy
    
    # Exact Match
    em = exact_match(pred_sql="SELECT * FROM students", gold_sql="SELECT * FROM students")
    print(f"Exact Match: {em}")  # True
    
    # Execution Accuracy
    ex = execution_accuracy(pred_result=[...], gold_result=[...])
    print(f"Execution Accuracy: {ex}")  # True
"""

import re
from typing import List, Dict, Any, Optional, Set


def normalize_sql(sql: str) -> str:
    """
    标准化 SQL（用于 Exact Match 计算）
    
    处理：
    1. 转小写
    2. 移除多余空白
    3. 移除注释
    4. 标准化引号
    
    Args:
        sql: 原始 SQL
    
    Returns:
        标准化后的 SQL
    """
    # 1. 转小写
    normalized = sql.lower()
    
    # 2. 移除注释（-- 和 /* */）
    normalized = re.sub(r"--.*$", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"/\*.*?\*/", "", normalized, flags=re.DOTALL)
    
    # 3. 标准化空白（多个空格/换行 -> 单个空格）
    normalized = re.sub(r"\s+", " ", normalized)
    
    # 4. 移除前后空白
    normalized = normalized.strip()
    
    # 5. 标准化引号（双引号 -> 单引号）
    normalized = normalized.replace('"', "'")
    
    return normalized


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """
    Exact Match (EM) 指标
    
    Args:
        pred_sql: 预测的 SQL
        gold_sql: 标准的 SQL（Ground Truth）
    
    Returns:
        是否完全匹配
    """
    pred_norm = normalize_sql(pred_sql)
    gold_norm = normalize_sql(gold_sql)
    
    return pred_norm == gold_norm


def execution_accuracy(
    pred_result: List[Dict],
    gold_result: List[Dict],
    ignore_order: bool = True
) -> bool:
    """
    Execution Accuracy (EX) 指标
    
    比较执行结果是否相同
    
    Args:
        pred_result: 预测的 SQL 执行结果（列表 of dict）
        gold_result: 标准的 SQL 执行结果
        ignore_order: 是否忽略行顺序
    
    Returns:
        执行结果是否相同
    """
    if len(pred_result) != len(gold_result):
        return False
    
    if ignore_order:
        # 忽略顺序：转换为集合比较
        pred_set = set(tuple(sort_dict(row).items()) for row in pred_result)
        gold_set = set(tuple(sort_dict(row).items()) for row in gold_result)
        return pred_set == gold_set
    else:
        # 考虑顺序：逐行比较
        for pred_row, gold_row in zip(pred_result, gold_result):
            if sort_dict(pred_row) != sort_dict(gold_row):
                return False
        return True


def sort_dict(d: Dict) -> Dict:
    """字典排序（用于比较）"""
    return dict(sorted(d.items()))


def clause_accuracy(
    pred_sql: str,
    gold_sql: str,
    clauses: List[str] = None
) -> Dict[str, bool]:
    """
    Clause-level Accuracy（Clause 级准确率）
    
    检查每个 clause 是否正确
    
    Args:
        pred_sql: 预测的 SQL
        gold_sql: 标准的 SQL
        clauses: 要检查的 clause 列表（默认：["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "LIMIT"]）
    
    Returns:
        每个 clause 的准确率字典
    """
    if clauses is None:
        clauses = ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT"]
    
    pred_norm = normalize_sql(pred_sql)
    gold_norm = normalize_sql(gold_sql)
    
    results = {}
    
    for clause in clauses:
        # 提取 clause 内容
        pred_clause = extract_clause(pred_norm, clause)
        gold_clause = extract_clause(gold_norm, clause)
        
        # 比较
        if pred_clause is None and gold_clause is None:
            results[clause] = True  # 两者都没有此 clause
        elif pred_clause is None or gold_clause is None:
            results[clause] = False  # 一个有一个没有
        else:
            results[clause] = (pred_clause == gold_clause)
    
    return results


def extract_clause(sql: str, clause: str) -> Optional[str]:
    """
    从 SQL 中提取指定 clause 的内容
    
    Args:
        sql: 标准化后的 SQL
        clause: clause 名称（如 "WHERE"）
    
    Returns:
        clause 内容，如果不存在则返回 None
    """
    sql_upper = sql.upper()
    clause_upper = clause.upper()
    
    if clause_upper not in sql_upper:
        return None
    
    # 找到 clause 起始位置
    start_idx = sql_upper.index(clause_upper)
    clause_content = sql[start_idx + len(clause_upper):].strip()
    
    # 找到 clause 结束位置（下一个 clause 或字符串结束）
    next_clauses = ["FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "JOIN"]
    
    for next_clause in next_clauses:
        if next_clause == clause_upper:
            continue  # 跳过自己
        
        idx = clause_content.upper().find(next_clause)
        if idx != -1:
            clause_content = clause_content[:idx].strip()
            break
    
    return clause_content if clause_content else None


def correction_success_rate(correction_history: List[Dict]) -> float:
    """
    纠错成功率
    
    Args:
        correction_history: 纠错历史列表，每个元素包含：
            - "turn_id": 轮次
            - "success": 是否纠错成功（bool）
    
    Returns:
        纠错成功率（0-1）
    """
    if not correction_history:
        return 0.0
    
    success_count = sum(1 for item in correction_history if item.get("success", False))
    return success_count / len(correction_history)


def average_turns(dialogue_history: List[Dict]) -> float:
    """
    平均对话轮数
    
    Args:
        dialogue_history: 对话历史列表
    
    Returns:
        平均轮数
    """
    if not dialogue_history:
        return 0.0
    
    return len(dialogue_history) / len(set(turn.get("session_id", 0) for turn in dialogue_history))


def component_based_accuracy(
    pred_sql: str,
    gold_sql: str
) -> Dict[str, float]:
    """
    基于组件的准确率（Spider 官方指标）
    
    检查 SQL 的各个组件是否正确：
    - SELECT clause
    - FROM clause (tables)
    - WHERE clause (conditions)
    - GROUP BY clause
    - ORDER BY clause
    - HAVING clause
    
    Args:
        pred_sql: 预测的 SQL
        gold_sql: 标准的 SQL
    
    Returns:
        每个组件的准确率（0 or 1）
    """
    # 简化版：使用 clause_accuracy
    return clause_accuracy(pred_sql, gold_sql)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("评估指标测试")
    print("=" * 60)
    
    # 测试 1：Exact Match
    print("\n[测试 1] Exact Match")
    pred = "SELECT * FROM students WHERE major = 'CS'"
    gold = "SELECT * FROM students WHERE major = 'CS'"
    em = exact_match(pred, gold)
    print(f"  Pred: {pred}")
    print(f"  Gold: {gold}")
    print(f"  Exact Match: {em}")  # True
    
    # 测试 2：Execution Accuracy
    print("\n[测试 2] Execution Accuracy")
    pred_result = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    gold_result = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    ex = execution_accuracy(pred_result, gold_result)
    print(f"  Pred Result: {pred_result}")
    print(f"  Gold Result: {gold_result}")
    print(f"  Execution Accuracy: {ex}")  # True
    
    # 测试 3：Clause Accuracy
    print("\n[测试 3] Clause Accuracy")
    pred = "SELECT name FROM students WHERE age > 18 GROUP BY major"
    gold = "SELECT name FROM students WHERE age > 18 GROUP BY major"
    ca = clause_accuracy(pred, gold)
    print(f"  Pred: {pred}")
    print(f"  Gold: {gold}")
    print(f"  Clause Accuracy:")
    for clause, acc in ca.items():
        print(f"    {clause}: {acc}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
