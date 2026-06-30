"""
ReAct Agent 核心模块 V2
实现 ReAct (Reasoning + Acting) 循环，支持本地小模型和远程大模型

架构：
- 小模型（Llama-3.1-8B + LoRA）：本地运行，低成本决策
- 大模型（智谱 AI API）：远程调用，高质量生成

使用方式：
    from src.agent.react_agent_v2 import ReactAgentV2
    
    # 使用 API 大模型（默认）
    agent = ReactAgentV2(db_id="student_db", use_small_model=False)
    
    # 使用本地小模型
    agent = ReactAgentV2(db_id="student_db", use_small_model=True)
    
    result = agent.run("列出所有计算机科学专业的学生")
    print(result["sql"])
    print(result["result"])
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

logger = get_logger(__name__)


class AgentState(Enum):
    """Agent 状态"""
    THINKING = "thinking"      # 思考中
    GENERATING = "generating"  # 生成 SQL 中
    EXECUTING = "executing"    # 执行 SQL 中
    CORRECTING = "correcting"   # 纠错中
    FINISHED = "finished"       # 完成


class ReactAgentV2:
    """
    ReAct Agent 核心类 V2
    实现 Reasoning + Acting 循环，支持模型选择
    """
    
    def __init__(
        self,
        db_id: str,
        use_small_model: bool = False,
        max_iterations: int = 10,
        temperature: float = 0.1
    ):
        """
        初始化 Agent
        
        Args:
            db_id: 数据库 ID
            use_small_model: 是否使用本地小模型（True：Llama-3.1-8B，False：智谱 AI API）
            max_iterations: 最大推理轮数
            temperature: 生成温度
        """
        self.db_id = db_id
        self.use_small_model = use_small_model
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # 初始化 SQL 生成器
        from src.agent.sql_generator_v2 import SQLGeneratorV2
        self.sql_generator = SQLGeneratorV2(use_small_model=use_small_model)
        
        # 初始化 SQL 执行器
        self.executor = SQLExecutor.from_db_id(db_id)
        
        # 获取数据库 Schema
        self.schema = self.executor.get_schema()
        
        # 状态记录
        self.state_history: List[Dict] = []
        self.generated_sql: Optional[str] = None
        self.execution_result: Optional[Dict] = None
        self.correction_history: List[Dict] = []
        
        model_type = "本地小模型 (Llama-3.1-8B + LoRA)" if use_small_model else "远程大模型 (智谱 AI API)"
        logger.info(f"ReactAgentV2 初始化完成 | 数据库: {db_id} | 模型: {model_type}")
    
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
                
                gen_result = self.sql_generator.generate_sql(
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
                state_record["result"] = f"生成 SQL: {self.generated_sql[:100]}..."
                logger.info(f"生成 SQL: {self.generated_sql}")
            
            # ===== Step 2: 执行 SQL =====
            logger.info("Action: 执行 SQL")
            state_record["action"] = "EXECUTE_SQL"
            
            exec_result = self.executor.execute(self.generated_sql)
            
            if exec_result.success:
                # 执行成功
                logger.info(f"SQL 执行成功，返回 {exec_result.rows} 行")
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
                state_record["result"] = f"执行失败: {exec_result.error[:100]}..."
                
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
                    
                    correct_result = self.sql_generator.correct_sql(
                        wrong_sql=self.generated_sql,
                        error_message=exec_result.error,
                        schema=self.schema,
                        question=question,
                        temperature=self.temperature
                    )
                    
                    if correct_result["success"]:
                        self.generated_sql = correct_result["sql"]
                        state_record["result"] += f" → 纠正为: {self.generated_sql[:100]}..."
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
            "correction_history": self.correction_history,
            "model_type": "small" if self.use_small_model else "large"
        }
        return response
    
    def switch_model(self, use_small_model: bool):
        """
        切换模型
        
        Args:
            use_small_model: 是否使用本地小模型
        """
        if use_small_model == self.use_small_model:
            logger.info(f"模型未变更（当前已使用 {'小模型' if use_small_model else '大模型'}）")
            return
        
        logger.info(f"切换模型：{'小模型' if use_small_model else '大模型'}")
        self.use_small_model = use_small_model
        self.sql_generator.switch_model(use_small_model)


# 导出
__all__ = ["ReactAgentV2", "AgentState"]


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="ReAct Agent V2 测试")
    parser.add_argument("--db_id", type=str, default="student_db", help="数据库 ID")
    parser.add_argument("--use_small_model", action="store_true", help="是否使用本地小模型")
    parser.add_argument("--question", type=str, default="列出所有计算机科学专业的学生", help="测试问题")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ReAct Agent V2 测试")
    print("=" * 60)
    print(f"数据库: {args.db_id}")
    print(f"模型: {'本地小模型 (Llama-3.1-8B)' if args.use_small_model else '远程大模型 (智谱 AI API)'}")
    print(f"问题: {args.question}")
    print("-" * 60)
    
    # 初始化 Agent
    agent = ReactAgentV2(
        db_id=args.db_id,
        use_small_model=args.use_small_model,
        max_iterations=5,
        temperature=0.1
    )
    
    # 运行
    result = agent.run(args.question)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("执行结果：")
    print("=" * 60)
    print(f"成功: {result['success']}")
    print(f"\n生成的 SQL:")
    print(result['sql'])
    
    if result['success']:
        print(f"\n执行结果（{result['result']['row_count']} 行）:")
        for row in result['result']['rows'][:10]:  # 只显示前 10 行
            print(row)
        if result['result']['row_count'] > 10:
            print(f"... (共 {result['result']['row_count']} 行)")
    
    if not result['success']:
        print(f"\n错误: {result['error']}")
    
    print(f"\n推理轨迹（{len(result['trace'])} 轮）:")
    for step in result['trace']:
        print(f"  第 {step['iteration']} 轮: {step['action']} -> {step['result'][:50]}...")
