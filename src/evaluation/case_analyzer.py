"""
Case 分析器模块

功能：
1. 分析错误样本
2. 错误分类（Schema 理解错误、Clause 错误、等）
3. 生成错误分布报告
4. 提供改进建议

使用示例：
    from src.evaluation.case_analyzer import CaseAnalyzer
    
    analyzer = CaseAnalyzer()
    analyzer.load_results("results/spider_dev_api.json")
    analyzer.analyze()
    analyzer.print_report()
    analyzer.save_report("reports/spider_dev_api_analysis.json")
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CaseAnalyzer:
    """
    Case 分析器
    
    职责：
    1. 加载评估结果
    2. 分析错误模式
    3. 分类错误
    4. 生成报告
    """
    
    def __init__(self):
        """初始化分析器"""
        self.results = None
        self.analysis = None
        logger.info("Case 分析器初始化完成")
    
    def load_results(self, results_path: str):
        """
        加载评估结果
        
        Args:
            results_path: 评估结果 JSON 文件路径
        """
        path = Path(results_path)
        if not path.exists():
            logger.error(f"结果文件不存在：{results_path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        logger.info(f"评估结果已加载（样本数：{len(self.results.get('results', []))}）")
    
    def analyze(self):
        """分析错误模式"""
        if self.results is None:
            logger.error("请先加载评估结果")
            return
        
        results = self.results.get("results", [])
        
        # 统计
        total = len(results)
        success = sum(1 for r in results if r.get("success", False))
        failed = total - success
        
        # 错误分类
        error_types = Counter()
        clause_errors = Counter()
        
        for r in results:
            if not r.get("success", False):
                # 分类错误
                error_type = self._classify_error(r)
                error_types[error_type] += 1
                
                # Clause 级错误
                if "clause" in r:
                    clause_errors[r["clause"]] += 1
        
        self.analysis = {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0,
            "error_types": dict(error_types),
            "clause_errors": dict(clause_errors)
        }
        
        logger.info(f"分析完成！成功率：{self.analysis['success_rate']:.2%}")
    
    def _classify_error(self, result: Dict) -> str:
        """
        错误分类
        
        分类：
        - schema_error: Schema 理解错误
        - clause_error: Clause 错误（WHERE, JOIN, etc.）
        - syntax_error: 语法错误
        - logic_error: 逻辑错误
        - execution_error: 执行错误
        - unknown: 未知错误
        """
        # 简化：基于错误信息分类
        if "error" in result:
            error_msg = result["error"].lower()
            
            if "schema" in error_msg or "table" in error_msg or "column" in error_msg:
                return "schema_error"
            elif "syntax" in error_msg:
                return "syntax_error"
            elif "execution" in error_msg:
                return "execution_error"
            else:
                return "unknown"
        
        # 如果没有错误信息，但基于 SQL 分析
        pred_sql = result.get("pred_sql", "")
        gold_sql = result.get("gold_sql", "")
        
        if not pred_sql:
            return "generation_failed"
        
        # 简化：默认为 logic_error
        return "logic_error"
    
    def print_report(self):
        """打印分析报告"""
        if self.analysis is None:
            print("请先运行 analyze()")
            return
        
        print("=" * 60)
        print("Case 分析报告")
        print("=" * 60)
        
        print(f"\n样本数：{self.analysis['total']}")
        print(f"成功：{self.analysis['success']}")
        print(f"失败：{self.analysis['failed']}")
        print(f"成功率：{self.analysis['success_rate']:.2%}")
        
        if self.analysis['error_types']:
            print(f"\n错误类型分布：")
            for error_type, count in self.analysis['error_types'].items():
                print(f"  {error_type}: {count} ({count/self.analysis['failed']:.1%})")
        
        if self.analysis['clause_errors']:
            print(f"\nClause 错误分布：")
            for clause, count in self.analysis['clause_errors'].items():
                print(f"  {clause}: {count}")
        
        print()
    
    def save_report(self, output_path: Optional[str] = None):
        """
        保存分析报告
        
        Args:
            output_path: 输出路径
        """
        if self.analysis is None:
            logger.error("请先运行 analyze()")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = project_root / "reports" / f"analysis_{timestamp}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析报告已保存 to {output_path}")


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Case 分析器测试")
    print("=" * 60)
    
    analyzer = CaseAnalyzer()
    
    # 测试：加载结果并分析
    print("\n提示：需要先运行评估生成结果文件")
    print("  示例：")
    print("    analyzer = CaseAnalyzer()")
    print("    analyzer.load_results('results/spider_dev_api.json')")
    print("    analyzer.analyze()")
    print("    analyzer.print_report()")
    
    print("\n" + "=" * 60)
    print("测试完成！")
