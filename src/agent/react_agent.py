"""
ReAct Agent 核心模块
实现 ReAct (Reasoning + Acting) 循环
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.helpers import format_prompt, clean_sql
from src.execution.sql_executor import SQLExecutor
from src.agent.sql_generator import generate_sql, correct_sql

logger = get_logger(__name__)


class AgentState(Enum):
    """Agent 状态"""
    THINKING = "thinking"      # 思考中
    GENERATING = "generating"  # 生成 SQL 中
    EXECUTING = "executing"    # 执行 SQL 中
    CORRECTING = "correcting"   # 纠错中
    FINISHED = "finished"       # 完成


class ReactAgent:
    """
    ReAct Agent 核心类
    实现 Reasoning + Acting 循环
    """
    
    def __init__(
        self,
        db_id: str,
        max_iterations: int = 10,
        temperature: float = 0.0
    ):
        """
        初始化 Agent
        
        Args:
            db_id: 数据库 ID
            max_iterations: 最大推理轮数
            temperature: 生成温度
        """
        self.db_id = db_id
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # 初始化 SQL 执行器
        self.executor = SQLExecutor.from_db_id(db_id)
        
        # 获取数据库 Schema
        self.schema = self.executor.get_schema()
        
        # 状态记录
        self.state_history: List[Dict] = []
        self.generated_sql: Optional[str] = None
        self.execution_result: Optional[Dict] = None
        self.correction_history: List[Dict] = []
        
        logger.info(f"ReactAgent 初始化完成，数据库: {db_id}")
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        运行 ReAct 循环
        
        Args:
            question: 用户问题（自然语言）
            
        Returns:
            包含 sql, result, success, trace 的字典
        """
        logger.info(f"开始处理问题: {question}")
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"=== 第 {iteration} 轮推理 ===")
            
            # 记录状态
            state_record = {
                "iteration": iteration,
                "action": None,
                "result": None
            }
            
            # ===== Step 1: 生成 SQL（第一轮）或纠错（后续轮） =====
            if iteration == 1 or self.generated_sql is None:
                # 生成初始 SQL
                logger.info("Action: 生成 SQL")
                state_record["action"] = "GENERATE_SQL"
                
                gen_result = generate_sql(
                    question=question,
                    schema=self.schema,
                    db_id=self.db_id,
                    temperature=self.temperature
                )
                
                if not gen_result["success"]:
                    logger.error(f"SQL 生成失败: {gen_result['error']}")
                    return self._build_response(
                        success=False,
                        error=f"SQL 生成失败: {gen_result['error']}",
                        trace=self.state_history
                    )
                
                self.generated_sql = gen_result["sql"]
                state_record["result"] = f"生成 SQL: {self.generated_sql}"
                logger.info(f"生成 SQL: {self.generated_sql}")
            
            # ===== Step 2: 执行 SQL =====
            logger.info("Action: 执行 SQL")
            state_record["action"] = "EXECUTE_SQL"
            
            exec_result = self.executor.execute(self.generated_sql)
            
            if exec_result.success:
                # 执行成功
                logger.info(f"SQL 执行成功，返回 {exec_result.row_count} 行")
                state_record["result"] = f"执行成功，返回 {exec_result.row_count} 行"
                
                self.execution_result = {
                    "success": True,
                    "rows": exec_result.rows,
                    "columns": exec_result.columns,
                    "row_count": exec_result.row_count,
                    "execution_time": exec_result.execution_time
                }
                
                # 成功，停止循环
                self.state_history.append(state_record)
                return self._build_response(
                    success=True,
                    sql=self.generated_sql,
                    result=self.execution_result,
                    trace=self.state_history
                )
            else:
                # 执行失败，需要纠错
                logger.warning(f"SQL 执行失败: {exec_result.error}")
                state_record["result"] = f"执行失败: {exec_result.error}"
                
                # 记录纠错历史
                self.correction_history.append({
                    "iteration": iteration,
                    "wrong_sql": self.generated_sql,
                    "error": exec_result.error
                })
                
                # 如果还有轮次，尝试纠错
                if iteration < self.max_iterations:
                    logger.info("Action: 纠正 SQL")
                    state_record["action"] = "CORRECT_SQL"
                    
                    correct_result = correct_sql(
                        wrong_sql=self.generated_sql,
                        error_message=exec_result.error,
                        schema=self.schema,
                        question=question,
                        temperature=self.temperature
                    )
                    
                    if correct_result["success"]:
                        self.generated_sql = correct_result["sql"]
                        state_record["result"] += f" → 纠正为: {self.generated_sql}"
                        logger.info(f"纠正后 SQL: {self.generated_sql}")
                    else:
                        logger.error(f"SQL 纠正失败: {correct_result['error']}")
                        # 继续下一轮（可能大模型能生成新的 SQL）
                        self.generated_sql = None
                
            self.state_history.append(state_record)
        
        # 达到最大轮数
        logger.warning(f"达到最大推理轮数 ({self.max_iterations})")
        return self._build_response(
            success=False,
            sql=self.generated_sql,
            error="达到最大推理轮数，仍未生成正确的 SQL",
            trace=self.state_history
        )
    
    def _build_response(
        self,
        success: bool,
        sql: Optional[str] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
        trace: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        构建响应字典
        """
        response = {
            "success": success,
            "sql": sql,
            "result": result,
            "error": error,
            "trace": trace or self.state_history,
            "correction_history": self.correction_history
        }
        return response


# 导出
__all__ = ["ReactAgent", "AgentState"]
