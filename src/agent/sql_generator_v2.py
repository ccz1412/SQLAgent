"""
SQL 生成器（支持本地小模型 + 远程大模型 API）

功能：
1. 生成 SQL（调用小模型或大模型）
2. 纠正 SQL（调用小模型或大模型）
3. 支持模型热切换

架构说明：
- 小模型（Llama-3.1-8B + LoRA）：本地运行，低成本
- 大模型（智谱 AI API）：远程调用，高质量

使用方式：
    from src.agent.sql_generator_v2 import SQLGeneratorV2
    
    generator = SQLGeneratorV2(use_small_model=True)  # 使用本地小模型
    # 或
    generator = SQLGeneratorV2(use_small_model=False)  # 使用 API 大模型
    
    # 生成 SQL
    result = generator.generate_sql(question="列出所有学生", schema="...")
    print(result["sql"])
    
    # 纠正 SQL
    result = generator.correct_sql(
        wrong_sql="SELECT * FROM students WHERE",
        error_message="incomplete SQL",
        schema="...",
        question="列出所有学生"
    )
    print(result["sql"])
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.helpers import format_prompt, clean_sql

logger = get_logger(__name__)


class SQLGeneratorV2:
    """
    SQL 生成器 V2（支持本地小模型 + API 大模型）
    
    职责：
    1. 封装本地小模型和远程大模型的调用
    2. 提供统一的 generate_sql() 和 correct_sql() 接口
    3. 支持模型热切换
    """
    
    def __init__(self, use_small_model: bool = False, small_model_config: Optional[Dict] = None):
        """
        初始化 SQL 生成器
        
        Args:
            use_small_model: 是否使用本地小模型（True：本地 Llama-3.1-8B，False：API 智谱 AI）
            small_model_config: 小模型配置（可选，覆盖默认配置）
        """
        self.use_small_model = use_small_model
        
        if use_small_model:
            # 加载本地小模型
            logger.info("正在加载本地小模型（Llama-3.1-8B + LoRA）...")
            from src.models.model_loader import get_model, ModelType
            self.small_model = get_model(ModelType.SMALL)
            self.large_model = None
            logger.info("本地小模型加载完成")
        else:
            # 使用 API 大模型
            logger.info("正在初始化 API 大模型客户端（智谱 AI）...")
            from src.models.model_loader import get_model, ModelType
            self.large_model = get_model(ModelType.LARGE)
            self.small_model = None
            logger.info("API 大模型客户端初始化完成")
    
    def generate_sql(
        self,
        question: str,
        schema: str,
        db_id: str,
        temperature: float = 0.1,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """
        生成 SQL
        
        Args:
            question: 用户问题（自然语言）
            schema: 数据库 schema
            db_id: 数据库 ID
            temperature: 生成温度
            few_shot_examples: Few-shot 示例（可选）
        
        Returns:
            {
                "success": True/False,
                "sql": "生成的 SQL",
                "error": "错误信息（如果有）"
            }
        """
        logger.info(f"生成 SQL | 问题：{question[:50]}...")
        
        if self.use_small_model and self.small_model:
            # 使用本地小模型生成 SQL
            return self._generate_sql_with_small_model(question, schema, temperature)
        else:
            # 使用 API 大模型生成 SQL
            return self._generate_sql_with_large_model(question, schema, temperature)
    
    def _generate_sql_with_small_model(
        self,
        question: str,
        schema: str,
        temperature: float
    ) -> Dict[str, any]:
        """使用本地小模型生成 SQL"""
        # 构造提示（Llama 3.1 Instruct 格式）
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是一个 SQL 生成专家。根据用户的自然语言问题，生成对应的 SQL 查询语句。

规则：
1. 只输出 SQL 语句，不要有任何解释或前缀
2. 使用标准的 SQL 语法
3. 确保 SQL 可以在 SQLite 中执行
4. 如果问题不明确，生成最合理的 SQL

<|eot_id|><|start_header_id|>user<|end_header_id|>

数据库 Schema：
{schema}

问题：{question}

请生成 SQL 查询语句：<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        try:
            sql = self.small_model.generate(
                prompt=prompt,
                max_tokens=512,
                temperature=temperature,
                do_sample=(temperature > 0)
            )
            
            # 清理 SQL
            sql = clean_sql(sql)
            
            logger.info(f"小模型生成 SQL：{sql[:100]}...")
            return {"success": True, "sql": sql, "error": None}
        
        except Exception as e:
            logger.error(f"小模型生成 SQL 失败：{e}")
            return {"success": False, "sql": None, "error": str(e)}
    
    def _generate_sql_with_large_model(
        self,
        question: str,
        schema: str,
        temperature: float
    ) -> Dict[str, any]:
        """使用 API 大模型生成 SQL"""
        try:
            sql = self.large_model.generate_sql(
                question=question,
                schema=schema,
                db_id="",
                few_shot_examples=None
            )
            
            # 清理 SQL
            sql = clean_sql(sql)
            
            logger.info(f"大模型生成 SQL：{sql[:100]}...")
            return {"success": True, "sql": sql, "error": None}
        
        except Exception as e:
            logger.error(f"大模型生成 SQL 失败：{e}")
            return {"success": False, "sql": None, "error": str(e)}
    
    def correct_sql(
        self,
        wrong_sql: str,
        error_message: str,
        schema: str,
        question: str,
        temperature: float = 0.1
    ) -> Dict[str, any]:
        """
        纠正 SQL
        
        Args:
            wrong_sql: 错误的 SQL
            error_message: 错误信息
            schema: 数据库 schema
            question: 原始问题
            temperature: 生成温度
        
        Returns:
            {
                "success": True/False,
                "sql": "纠正后的 SQL",
                "error": "错误信息（如果有）"
            }
        """
        logger.info(f"纠正 SQL | 错误：{error_message[:50]}...")
        
        if self.use_small_model and self.small_model:
            # 使用本地小模型纠正 SQL
            return self._correct_sql_with_small_model(
                wrong_sql, error_message, schema, question, temperature
            )
        else:
            # 使用 API 大模型纠正 SQL
            return self._correct_sql_with_large_model(
                wrong_sql, error_message, schema, question, temperature
            )
    
    def _correct_sql_with_small_model(
        self,
        wrong_sql: str,
        error_message: str,
        schema: str,
        question: str,
        temperature: float
    ) -> Dict[str, any]:
        """使用本地小模型纠正 SQL"""
        # 构造 ReAct 风格的提示（引导小模型进行推理）
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是一个 SQL 纠错专家。根据错误信息，仔细分析并修正 SQL 查询语句。

请按以下步骤思考：
1. Thought：分析错误原因
2. Action：修正 SQL（只输出修正后的完整 SQL 语句）

规则：
- 只输出修正后的 SQL 语句
- 确保修正后的 SQL 可以正确执行
- 仔细核对表名、列名、语法

<|eot_id|><|start_header_id|>user<|end_header_id|>

数据库 Schema：
{schema}

原始问题：{question}

错误的 SQL：
{wrong_sql}

错误信息：
{error_message}

请修正 SQL：<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Thought: """
        
        try:
            response = self.small_model.generate(
                prompt=prompt,
                max_tokens=512,
                temperature=temperature,
                do_sample=(temperature > 0)
            )
            
            # 提取 SQL（从 Thought 之后的内容）
            sql = self._extract_sql_from_response(response)
            sql = clean_sql(sql)
            
            logger.info(f"小模型纠正 SQL：{sql[:100]}...")
            return {"success": True, "sql": sql, "error": None}
        
        except Exception as e:
            logger.error(f"小模型纠正 SQL 失败：{e}")
            return {"success": False, "sql": None, "error": str(e)}
    
    def _correct_sql_with_large_model(
        self,
        wrong_sql: str,
        error_message: str,
        schema: str,
        question: str,
        temperature: float
    ) -> Dict[str, any]:
        """使用 API 大模型纠正 SQL"""
        try:
            sql = self.large_model.correct_sql(
                wrong_sql=wrong_sql,
                error_message=error_message,
                schema=schema,
                question=question
            )
            
            # 清理 SQL
            sql = clean_sql(sql)
            
            logger.info(f"大模型纠正 SQL：{sql[:100]}...")
            return {"success": True, "sql": sql, "error": None}
        
        except Exception as e:
            logger.error(f"大模型纠正 SQL 失败：{e}")
            return {"success": False, "sql": None, "error": str(e)}
    
    def _extract_sql_from_response(self, response: str) -> str:
        """
        从模型响应中提取 SQL
        
        小模型可能会输出：
        - "Thought: ... Action: SQL 语句"
        - 直接输出 SQL
        """
        # 如果响应包含 "Action:"，提取其后的内容
        if "Action:" in response:
            sql_start = response.index("Action:") + len("Action:")
            sql = response[sql_start:].strip()
            return sql
        
        # 否则，假设整个响应都是 SQL
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
        
        if use_small_model:
            from src.models.model_loader import get_model, ModelType
            self.small_model = get_model(ModelType.SMALL)
            self.large_model = None
        else:
            from src.models.model_loader import get_model, ModelType
            self.large_model = get_model(ModelType.LARGE)
            self.small_model = None


# 保持向后兼容（默认使用 API 大模型）
def generate_sql(
    question: str,
    schema: str,
    db_id: str,
    temperature: float = 0.0
) -> Dict[str, any]:
    """
    生成 SQL（向后兼容函数）
    
    默认使用 API 大模型（智谱 AI）
    """
    generator = SQLGeneratorV2(use_small_model=False)
    return generator.generate_sql(question, schema, db_id, temperature)


def correct_sql(
    wrong_sql: str,
    error_message: str,
    schema: str,
    question: str,
    temperature: float = 0.0
) -> Dict[str, any]:
    """
    纠正 SQL（向后兼容函数）
    
    默认使用 API 大模型（智谱 AI）
    """
    generator = SQLGeneratorV2(use_small_model=False)
    return generator.correct_sql(wrong_sql, error_message, schema, question, temperature)


if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path
    
    # 添加项目根目录到 sys.path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 60)
    print("SQL 生成器 V2 测试")
    print("=" * 60)
    
    # 测试配置
    USE_SMALL_MODEL = False  # 改为 True 可测试本地小模型
    
    # 初始化生成器
    generator = SQLGeneratorV2(use_small_model=USE_SMALL_MODEL)
    
    # 测试生成 SQL
    print(f"\n{'-' * 60}")
    print("测试 1：生成 SQL")
    result = generator.generate_sql(
        question="列出所有计算机科学专业的学生",
        schema="Table: students (id, name, major, gpa)",
        db_id="test_db",
        temperature=0.1
    )
    print(f"成功：{result['success']}")
    if result['success']:
        print(f"SQL：{result['sql']}")
    else:
        print(f"错误：{result['error']}")
    
    print(f"\n{'=' * 60}")
    print("测试完成！")
