"""
评估模块包

包含：
- metrics.py: 评估指标（Exact Match, Execution Accuracy, etc.）
- evaluator.py: 评估器（运行评估、生成报告）
- case_analyzer.py: Case 分析器（错误分类、趋势分析）
"""

from src.evaluation.metrics import exact_match, execution_accuracy, clause_accuracy
from src.evaluation.evaluator import Evaluator
from src.evaluation.case_analyzer import CaseAnalyzer

__all__ = [
    "exact_match",
    "execution_accuracy",
    "clause_accuracy",
    "Evaluator",
    "CaseAnalyzer"
]
