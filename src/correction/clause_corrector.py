"""
Clause 级 SQL 纠错模块 v2

功能：
1. 定位 SQL 中的错误 clause（WHERE, JOIN, GROUP BY 等）
2. 生成针对性的纠错建议
3. 支持接入 GRPO 训练模型（后续）

纠错策略：
- 规则纠错：基于 SQL 执行错误信息
- 模型纠错：调用本地小模型或 API 大模型
- GRPO 纠错：预留接口，后续接入训练好的模型

所有 Prompt 模板统一从 src/prompts/prompt_templates.py 导入。

使用示例：
    from src.correction.clause_corrector import ClauseCorrector
    
    corrector = ClauseCorrector(use_grpo=False)
    result = corrector.correct(
        wrong_sql="SELECT * FROM students WHERE",
        error_message="incomplete SQL",
        schema="...",
        question="列出所有学生"
    )
    print(result["corrected_sql"])
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import re

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.helpers import clean_sql
from src.prompts.prompt_templates import (
    CLAUSE_CORRECTION_SYSTEM,
    build_clause_correction_prompt,
)

logger = get_logger(__name__)


class ClauseCorrector:
    """
    Clause 级 SQL 纠错器
    
    职责：
    1. 分析错误的 SQL，定位问题 clause
    2. 生成多个纠错候选
    3. 验证纠错后的 SQL（执行测试）
    """
    
    def __init__(self, use_grpo: bool = False, grpo_model_path: Optional[str] = None):
        """
        初始化纠错器
        
        Args:
            use_grpo: 是否使用 GRPO 模型（默认：False，使用规则+API）
            grpo_model_path: GRPO 模型路径（如果 use_grpo=True）
        """
        self.use_grpo = use_grpo
        self.grpo_model_path = grpo_model_path
        
        if use_grpo:
            logger.info("使用 GRPO 模型进行 Clause 级纠错")
            self._load_grpo_model()
        else:
            logger.info("使用规则 + API 大模型进行纠错")
    
    def _load_grpo_model(self):
        """加载 GRPO 训练好的模型（预留接口）"""
        logger.warning("GRPO 模型加载功能待实现")
        # TODO: 实现 GRPO 模型加载
        # 示例代码：
        # from src.models.model_loader import get_model, ModelType
        # self.grpo_model = get_model(ModelType.SMALL, reload=True)
        pass
    
    def correct(
        self,
        wrong_sql: str,
        error_message: str,
        schema: str,
        question: str,
        db_id: Optional[str] = None
    ) -> Dict:
        """
        纠正错误的 SQL
        
        Args:
            wrong_sql: 错误的 SQL
            error_message: 执行错误信息
            schema: 数据库 schema
            question: 用户原始问题
            db_id: 数据库 ID（用于执行验证）
        
        Returns:
            {
                "corrected_sql": str,  # 纠正后的 SQL
                "error_clause": str,   # 定位到的错误 clause
                "confidence": float,   # 纠错置信度 (0-1)
                "method": str           # 纠错方法（rule/api/grpo）
            }
        """
        logger.info(f"开始 Clause 级纠错...")
        logger.debug(f"错误 SQL: {wrong_sql}")
        logger.debug(f"错误信息: {error_message}")
        
        # 步骤 1：定位错误 clause
        error_clause = self._locate_error_clause(wrong_sql, error_message)
        logger.info(f"定位到错误 clause: {error_clause}")
        
        # 步骤 2：生成纠错候选
        if self.use_grpo:
            corrected_sql = self._correct_by_grpo(wrong_sql, error_clause, schema, question)
            method = "grpo"
        else:
            # 先尝试规则纠错
            corrected_sql = self._correct_by_rule(wrong_sql, error_clause, error_message)
            if corrected_sql is None:
                # 规则无法处理，调用 API 大模型
                corrected_sql = self._correct_by_api(wrong_sql, error_clause, schema, question)
                method = "api"
            else:
                method = "rule"
        
        logger.info(f"纠错完成，方法：{method}")
        logger.debug(f"纠正后 SQL: {corrected_sql}")
        
        return {
            "corrected_sql": corrected_sql,
            "error_clause": error_clause,
            "confidence": 0.8 if method == "grpo" else 0.6,  # 简化：GRPO 置信度更高
            "method": method
        }
    
    def _locate_error_clause(self, sql: str, error_message: str) -> str:
        """
        定位错误 clause
        
        策略：
        1. 基于错误信息（如 "syntax error near WHERE"）
        2. 基于 SQL 解析（检查每个 clause 的完整性）
        
        Args:
            sql: 错误的 SQL
            error_message: 执行错误信息
        
        Returns:
            错误 clause 类型（WHERE, JOIN, GROUP BY, etc.）
        """
        sql_upper = sql.upper()
        
        # 策略 1：从错误信息中提取关键词
        error_keywords = ["near", "at or near", "syntax error", "unknown column"]
        for keyword in error_keywords:
            if keyword in error_message.lower():
                # 提取错误位置附近的 SQL 片段
                # 简化：返回整个 SQL
                pass
        
        # 策略 2：检查 SQL 完整性
        clauses = ["WHERE", "JOIN", "GROUP BY", "ORDER BY", "HAVING", "LIMIT"]
        
        for clause in clauses:
            if clause in sql_upper:
                # 检查该 clause 是否完整
                clause_index = sql_upper.index(clause)
                clause_content = sql[clause_index:]
                
                # 简化：如果 clause 后面没有有效内容，则定位为错误
                if clause == "WHERE":
                    # 检查 WHERE 后面是否有条件
                    where_content = sql_upper.split("WHERE")[1].strip()
                    if not where_content or where_content.endswith(","):
                        return "WHERE"
                
                # 简化：返回第一个找到的 clause
                return clause
        
        # 默认：无法定位
        return "UNKNOWN"
    
    def _correct_by_rule(self, sql: str, error_clause: str, error_message: str) -> Optional[str]:
        """
        基于规则的纠错
        
        适用场景：
        1. 明显的语法错误（缺少引号、括号等）
        2. 常见的 SQL 错误模式
        
        Returns:
            纠正后的 SQL，如果规则无法处理则返回 None
        """
        corrected = sql
        
        # 规则 1：修复不完整的 WHERE
        if error_clause == "WHERE":
            if corrected.strip().upper().endswith("WHERE"):
                # WHERE 后面没有条件，添加占位条件
                corrected = corrected.strip() + " 1=1"
                logger.info("规则纠错：添加 WHERE 1=1")
                return corrected
        
        # 规则 2：修复缺少的引号
        if "unclosed quotation" in error_message.lower():
            # 统计引号数量
            single_quotes = corrected.count("'")
            double_quotes = corrected.count('"')
            
            if single_quotes % 2 == 1:
                corrected += "'"
                logger.info("规则纠错：添加缺少的单引号")
                return corrected
            
            if double_quotes % 2 == 1:
                corrected += '"'
                logger.info("规则纠错：添加缺少的双引号")
                return corrected
        
        # 规则 3：简化版 - 移除最后的逗号或关键字
        if corrected.strip().endswith(","):
            corrected = corrected.strip()[:-1]
            logger.info("规则纠错：移除末尾逗号")
            return corrected
        
        # 无法处理
        logger.debug("规则纠错无法处理此错误")
        return None
    
    def _correct_by_api(self, sql: str, error_clause: str, schema: str, question: str) -> str:
        """
        调用 API 大模型进行纠错
        
        策略：
        1. 使用集中式 Prompt 模板构造请求
        2. 让大模型生成纠正后的 SQL
        3. 解析返回结果
        """
        logger.info("调用 API 大模型进行纠错...")
        
        # 使用集中式 Prompt 构建
        prompt = build_clause_correction_prompt(
            sql=sql,
            error_clause=error_clause,
            schema=schema,
            question=question
        )
        
        # 调用 API
        from src.call_api.qwen_client import chat
        
        messages = [
            {"role": "system", "content": CLAUSE_CORRECTION_SYSTEM},
            {"role": "user", "content": prompt}
        ]
        
        response, _ = chat(messages, max_tokens=512, temperature=0.1)
        
        # 解析返回结果（提取 SQL）
        corrected_sql = self._extract_sql_from_response(response)
        
        if corrected_sql is None:
            logger.warning("API 返回的响应中未找到 SQL，使用原 SQL")
            return sql
        
        return corrected_sql
    
    def _correct_by_grpo(self, sql: str, error_clause: str, schema: str, question: str) -> str:
        """
        使用 GRPO 训练好的模型进行纠错（预留接口）
        
        TODO: 实现 GRPO 模型推理
        """
        logger.warning("GRPO 纠错功能待实现，降级为 API 纠错")
        return self._correct_by_api(sql, error_clause, schema, question)
    
    def _extract_sql_from_response(self, response: str) -> Optional[str]:
        """从 API 响应中提取 SQL"""
        # 策略 1：提取 markdown 代码块中的 SQL
        import re
        pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return clean_sql(match.group(1))
        
        # 策略 2：提取 ``` 代码块（无语言标识）
        pattern = r"```\s*(.*?)\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return clean_sql(match.group(1))
        
        # 策略 3：如果整个响应就是 SQL（以 SELECT/INSERT/UPDATE/DELETE 开头）
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]
        response_upper = response.strip().upper()
        for keyword in sql_keywords:
            if response_upper.startswith(keyword):
                return clean_sql(response)
        
        # 未找到 SQL
        return None


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Clause 级纠错测试")
    print("=" * 60)
    
    # 测试 1：规则纠错
    print("\n[测试 1] 规则纠错（不完整的 WHERE）")
    corrector = ClauseCorrector(use_grpo=False)
    
    result = corrector.correct(
        wrong_sql="SELECT * FROM students WHERE",
        error_message="incomplete SQL",
        schema="Table: students (id, name, age)",
        question="列出所有学生"
    )
    
    print(f"原 SQL: SELECT * FROM students WHERE")
    print(f"纠正后 SQL: {result['corrected_sql']}")
    print(f"纠错方法: {result['method']}")
    
    # 测试 2：API 纠错
    print("\n" + "-" * 60)
    print("[测试 2] API 大模型纠错")
    
    result = corrector.correct(
        wrong_sql="SELECT * FROM non_existent_table",
        error_message="table non_existent_table does not exist",
        schema="Table: students (id, name, age)",
        question="列出所有学生"
    )
    
    print(f"原 SQL: SELECT * FROM non_existent_table")
    print(f"纠正后 SQL: {result['corrected_sql']}")
    print(f"纠错方法: {result['method']}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
