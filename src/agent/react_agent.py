"""
ReAct Agent 核心模块
实现 Reasoning + Acting 循环（双模型纠错架构）

架构设计：
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  用户问题    │ ──→ │  生成LLM (API)   │ ──→ │  SQL执行器   │
│  (自然语言)   │     │  智谱AI glm-4    │     │  SQLite      │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                         │
                    ┌──────────────────┐              │
                    │  纠错LLM(本地)    │ ←──────────────┘
                    │  Llama-3.1-8B    │   执行结果+SQL
                    │  + LoRA 权重     │
                    └────────┬─────────┘
                             │
                    needs_correction?
                      ↓         ↓
                    是          否 → 返回结果
                      ↓
                  纠正后SQL → 重新执行

使用方式：
    from src.agent.react_agent import ReactAgent

    # 默认模式：API生成 + 本地Llama纠错
    agent = ReactAgent(db_id="student_db")
    result = agent.run("列出所有计算机科学专业的学生")

    # 仅API模式（不使用本地纠错模型，降级为API自纠错）
    agent = ReactAgent(db_id="student_db", use_correction_model=False)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.helpers import clean_sql
from src.execution.sql_executor import SQLExecutor
from src.agent.sql_generator import generate_sql, correct_sql as api_correct_sql

logger = get_logger(__name__)


class AgentState(Enum):
    """Agent 状态"""
    THINKING = "thinking"
    GENERATING = "generating"   # 生成SQL中（调用API）
    EXECUTING = "executing"     # 执行SQL中
    REVIEWING = "reviewing"      # 纠错LLM审查中
    CORRECTING = "correcting"    # 纠错中
    FINISHED = "finished"


class CorrectionReviewResult:
    """纠错LLM的审查结果"""
    def __init__(
        self,
        needs_correction: bool,
        reason: str = "",
        corrected_sql: Optional[str] = None,
        confidence: float = 0.0,
        error_clause: str = "",
        raw_response: str = ""
    ):
        self.needs_correction = needs_correction
        self.reason = reason
        self.corrected_sql = corrected_sql
        self.confidence = confidence
        self.error_clause = error_clause
        self.raw_response = raw_response

    def to_dict(self) -> Dict:
        return {
            "needs_correction": self.needs_correction,
            "reason": self.reason,
            "corrected_sql": self.corrected_sql,
            "confidence": self.confidence,
            "error_clause": self.error_clause
        }


class ReactAgent:
    """
    ReAct Agent 核心类

    实现双模型协作的推理循环：
    - 生成模型（API 大模型）：负责生成 SQL
    - 纠错模型（本地 Llama-3.1-8B + LoRA）：负责审查和纠正 SQL
    """

    def __init__(
        self,
        db_id: str,
        use_small_model: bool = False,
        use_correction_model: bool = True,
        max_iterations: int = 5,
        temperature: float = 0.1,
        correction_threshold: float = 0.3
    ):
        """
        初始化 Agent

        Args:
            db_id: 数据库 ID
            use_small_model: 是否使用本地小模型进行 SQL 生成（False=用API）
            use_correction_model: 是否使用本地纠错模型(Llama)进行审查（True=双模型模式）
            max_iterations: 最大推理轮数
            temperature: 生成温度
            correction_threshold: 纠错置信度阈值（低于此值则认为需要纠正）
        """
        self.db_id = db_id
        self.use_small_model = use_small_model
        self.use_correction_model = use_correction_model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.correction_threshold = correction_threshold

        # 初始化 SQL 执行器
        self.executor = SQLExecutor.from_db_id(db_id)
        self.schema = self.executor.get_schema()

        # 纠错模型实例（懒加载）
        self._correction_model = None

        # 状态记录
        self.state_history: List[Dict] = []
        self.generated_sql: Optional[str] = None
        self.execution_result: Optional[Dict] = None
        self.correction_history: List[Dict] = []

        gen_model = "本地Llama-3.1-8B" if use_small_model else "API(智谱AI)"
        corr_mode = f"+本地Llama纠错" if use_correction_model else "+API自纠错"
        logger.info(f"ReactAgent初始化完成 | DB:{db_id} | 生成:{gen_model} | {corr_mode}")

    @property
    def correction_model(self):
        """懒加载纠错模型（本地 Llama-3.1-8B + LoRA）"""
        if self._correction_model is None and self.use_correction_model:
            try:
                from src.models.model_loader import ModelLoader, ModelType
                loader = ModelLoader()
                self._correction_model = loader.get_model(ModelType.SMALL)
                logger.info("本地纠错模型加载成功")
            except Exception as e:
                logger.warning(f"本地纠错模型加载失败，降级为 API 自纠错: {e}")
                self.use_correction_model = False
        return self._correction_model

    def run(self, question: str) -> Dict[str, Any]:
        """
        运行 ReAct 双模型推理循环

        流程：
        第 N 轮：
          1. 生成SQL（API/小模型）
          2. 执行SQL
          3. 如果执行成功 → 让纠错LLM审查（可选但推荐）
             - 纠错LLM说OK → 返回结果
             - 纠错LLM说要修 → 应用修正 → 回到步骤2
          4. 如果执行失败 → 让纠错LLM纠正 → 回到步骤2

        Args:
            question: 用户问题（自然语言）

        Returns:
            包含 sql, result, success, trace 的字典
        """
        logger.info(f"开始处理问题: {question}")

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"=== 第 {iteration} 轮推理 ===")

            state_record = {
                "iteration": iteration,
                "action": None,
                "result": None
            }

            # ===== Step 1: 生成 SQL =====
            if iteration == 1 or self.generated_sql is None:
                state_record["action"] = "GENERATE_SQL"
                gen_result = generate_sql(
                    question=question,
                    schema=self.schema,
                    db_id=self.db_id,
                    temperature=self.temperature
                )

                if not gen_result["success"]:
                    logger.error(f"SQL生成失败: {gen_result['error']}")
                    return self._build_response(
                        success=False, error=f"SQL生成失败: {gen_result['error']}"
                    )

                self.generated_sql = gen_result["sql"]
                state_record["result"] = f"生成SQL: {self.generated_sql[:100]}..."
                logger.info(f"生成SQL: {self.generated_sql}")

            # ===== Step 2: 执行 SQL =====
            state_record["action"] = "EXECUTE_SQL"
            exec_result = self.executor.execute(self.generated_sql)

            if exec_result.success:
                # ===== 执行成功 → 纠错LLM审查（双模型核心）=====
                logger.info(f"SQL执行成功({exec_result.row_count}行)，进入审查阶段")

                review_result = self._review_with_correction_model(
                    sql=self.generated_sql,
                    exec_success=True,
                    error_message=None,
                    question=question
                )

                if review_result.needs_correction:
                    # 纠错模型认为需要修正
                    logger.info(f"纠错模型建议修正(原因: {review_result.reason})")
                    state_record["action"] = "CORRECT_BY_MODEL"

                    if review_result.corrected_sql:
                        self.generated_sql = review_result.corrected_sql
                        state_record["result"] += f" → 模型纠正: {self.generated_sql[:80]}..."

                        # 记录纠错历史
                        self.correction_history.append({
                            "iteration": iteration,
                            "wrong_sql": self.generated_sql,
                            "reason": review_result.reason,
                            "method": "local_model",
                            "trigger": "review_after_success"
                        })
                        # 不return，继续下一轮执行修正后的SQL
                    else:
                        # 模型说需要修正但没给出修正方案，直接返回当前结果
                        logger.warning("纠错模型要求修正但未提供修正SQL，接受当前结果")
                        return self._finalize_success(exec_result, state_record)
                else:
                    # 纠错模型认为没问题 → 返回结果
                    return self._finalize_success(exec_result, state_record)

            else:
                # ===== 执行失败 → 必须纠错 =====
                logger.warning(f"SQL执行失败: {exec_result.error}")
                state_record["result"] = f"执行失败: {exec_result.error[:80]}..."

                # 记录错误
                self.correction_history.append({
                    "iteration": iteration,
                    "wrong_sql": self.generated_sql,
                    "error": exec_result.error,
                    "method": None,
                    "trigger": "execution_failure"
                })

                if iteration < self.max_iterations:
                    state_record["action"] = "CORRECT_SQL"

                    # 使用纠错模型或API进行纠正
                    correct_result = self._perform_correction(
                        wrong_sql=self.generated_sql,
                        error_message=str(exec_result.error),
                        question=question
                    )

                    if correct_result and correct_result != self.generated_sql:
                        self.generated_sql = correct_result
                        state_record["result"] += f" → 纠正为: {correct_result[:80]}..."
                        logger.info(f"纠正后SQL: {correct_result}")
                    else:
                        # 纠正失败，清空让下一轮重新生成
                        logger.warning("纠正失败，下一轮将重新生成SQL")
                        self.generated_sql = None

            self.state_history.append(state_record)

        # 达到最大轮数
        logger.warning(f"达到最大推理轮数({self.max_iterations})")
        return self._build_response(
            success=False,
            sql=self.generated_sql,
            error="达到最大推理轮数，仍未得到正确结果"
        )

    def _review_with_correction_model(
        self,
        sql: str,
        exec_success: bool,
        error_message: Optional[str],
        question: str
    ) -> CorrectionReviewResult:
        """
        使用纠错模型审查 SQL

        这是双模型架构的核心方法。
        将生成的SQL交给本地微调过的Llama-3.1-8B模型，
        让它判断这个SQL是否需要纠正。

        Args:
            sql: 待审查的SQL
            exec_success: 是否执行成功
            error_message: 错误信息（如果失败）
            question: 原始用户问题

        Returns:
            CorrectionReviewResult
        """
        # 如果没有启用纠错模型，默认不纠正
        if not self.use_correction_model:
            return CorrectionReviewResult(
                needs_correction=False,
                reason="纠错模型未启用"
            )

        # 尝试获取纠错模型
        model = self.correction_model
        if model is None:
            # 加载失败，回退到不纠正
            return CorrectionReviewResult(
                needs_correction=False,
                reason="纠错模型不可用"
            )

        # 构造审查 Prompt（针对微调模型的输入格式）
        review_prompt = self._build_review_prompt(sql, exec_success, error_message, question)

        try:
            # 调用本地模型进行审查
            response = self._call_local_model(model, review_prompt)

            # 解析模型的判断结果
            parsed = self._parse_review_response(response, sql)

            return parsed

        except Exception as e:
            logger.error(f"纠错模型审查异常: {e}")
            return CorrectionReviewResult(
                needs_correction=False,
                reason=f"审查异常: {e}"
            )

    def _perform_correction(
        self,
        wrong_sql: str,
        error_message: str,
        question: str
    ) -> Optional[str]:
        """
        执行纠错操作

        优先使用本地纠错模型，降级为API纠错。

        Args:
            wrong_sql: 错误的SQL
            error_message: 错误信息
            question: 用户问题

        Returns:
            纠正后的SQL，如果无法纠正则返回None
        """
        # 尝试本地纠错模型
        if self.use_correction_model:
            model = self.correction_model
            if model is not None:
                try:
                    correct_prompt = self._build_correct_prompt(wrong_sql, error_message, question)
                    response = self._call_local_model(model, correct_prompt)
                    corrected = self._extract_sql_from_text(response)
                    if corrected and corrected != wrong_sql:
                        logger.info(f"本地模型纠错成功")
                        return corrected
                except Exception as e:
                    logger.warning(f"本地纠错模型失败: {e}，降级为API纠错")

        # 降级：API纠错
        try:
            result = api_correct_sql(
                wrong_sql=wrong_sql,
                error_message=error_message,
                schema=self.schema,
                question=question,
                temperature=self.temperature
            )
            if result.get("success") and result.get("sql"):
                logger.info("API纠错成功（降级）")
                return result["sql"]
        except Exception as e:
            logger.error(f"API纠错也失败了: {e}")

        return None

    def _build_review_prompt(
        self, sql: str, exec_success: bool, error_msg: Optional[str], question: str
    ) -> str:
        """构造给纠错模型的审查 Prompt"""
        exec_status = "成功执行" if exec_success else f"执行失败: {error_msg}"
        prompt = f"""[SQL审查任务]

数据库Schema:
{self.schema}

用户问题: {question}

待审查SQL: {sql}

执行状态: {exec_status}

请判断这条SQL是否需要纠正。输出格式(JSON):
{{"needs_correction": true/false, "reason": "...", "corrected_sql": "..."(如需要纠正), "confidence": 0.0-1.0}}

注意：
- 如果SQL能正确回答用户问题且无语法/逻辑错误，设needs_correction=false
- 如果SQL有潜在问题（即使执行成功了），也应指出并给出修正版
- confidence表示你的判断确信度"""
        return prompt

    def _build_correct_prompt(self, wrong_sql: str, error_msg: str, question: str) -> str:
        """构造给纠错模型的纠正 Prompt"""
        prompt = f"""[SQL纠正任务]

数据库Schema:
{self.schema}

用户问题: {question}

错误的SQL: {wrong_sql}
错误信息: {error_msg}

请纠正上述SQL，使其能正确执行并准确回答用户问题。只输出纠正后的SQL语句。"""
        return prompt

    def _call_local_model(self, model, prompt: str) -> str:
        """
        调用本地纠错模型（Llama-3.1-8B + LoRA）
        
        使用 SmallModel.generate() 接口，自动处理 tokenize 和 decode。
        
        Args:
            model: SmallModel 实例（来自 ModelLoader）
            prompt: 输入prompt
            
        Returns:
            模型生成的文本
        """
        try:
            # SmallModel.generate() 可直接接受 prompt 字符串
            # 如需使用聊天模板，可改用 generate_with_chat_template()
            response = model.generate(
                prompt=prompt,
                max_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True
            )
            return response.strip()
        except Exception as e:
            logger.error(f"调用本地模型失败: {e}")
            # 降级：尝试直接用 transformer 接口
            return self._call_local_model_raw(model, prompt)

    def _call_local_model_raw(self, model, prompt: str) -> str:
        """
        降级方案：直接使用 transformer 接口调用模型
        
        当 SmallModel.generate() 接口不可用时使用。
        """
        try:
            import torch
            # 使用 SmallModel 的 tokenizer 和 model 属性
            tokenizer = model.tokenizer
            llm_model = model.model
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode（只取新生成的 token）
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return response.strip()
        except Exception as e:
            logger.error(f"降级调用也失败: {e}")
            return "{}"  # 返回空JSON，让解析逻辑处理

    def _parse_review_response(self, response: str, original_sql: str) -> CorrectionReviewResult:
        """解析纠错模型的审查响应"""
        import re
        import json

        text = response.strip()

        # 尝试提取JSON
        json_match = re.search(r'\{[^{}]*"needs_correction"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                needs = data.get("needs_correction", False)
                # 处理各种可能的布尔值
                if isinstance(needs, str):
                    needs = needs.lower() in ("true", "yes", "1", "需要", "是")
                corrected = data.get("corrected_sql")

                return CorrectionReviewResult(
                    needs_correction=needs,
                    reason=data.get("reason", ""),
                    corrected_sql=clean_sql(corrected) if corrected else None,
                    confidence=float(data.get("confidence", 0.5)),
                    raw_response=response
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # 无法解析JSON，基于关键词判断
        needs_keywords = ["需要纠正", "有误", "错误", "incorrect", "wrong", "fix", "修改"]
        ok_keywords = ["正确", "无需纠正", "ok", "good", "correct", "没问题", "通过"]

        for kw in needs_keywords:
            if kw in text.lower():
                # 尝试从中提取修正后的SQL
                corrected = self._extract_sql_from_text(text)
                return CorrectionReviewResult(
                    needs_correction=True,
                    reason=f"模型响应包含'{kw}'",
                    corrected_sql=corrected,
                    raw_response=response
                )

        for kw in ok_keywords:
            if kw in text.lower():
                return CorrectionReviewResult(
                    needs_correction=False,
                    reason=f"模型响应确认OK('{kw}')",
                    raw_response=response
                )

        # 默认：不确定时不纠正（因为执行已经成功了）
        return CorrectionReviewResult(
            needs_correction=False,
            reason="无法解析模型响应，默认信任原SQL（已执行成功）",
            raw_response=response
        )

    def _extract_sql_from_text(self, text: str) -> Optional[str]:
        """从文本中提取SQL语句"""
        import re

        # 策略1: ```sql ... ```
        match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if match:
            return clean_sql(match.group(1))

        # 策略2: ``` ... ```
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            candidate = clean_sql(match.group(1))
            if candidate.upper().startswith(("SELECT", "WITH")):
                return candidate

        # 策略3: 直接查找SELECT语句
        lines = text.split('\n')
        sql_lines = []
        capturing = False
        for line in lines:
            if re.search(r'\bSELECT\b', line, re.IGNORECASE):
                capturing = True
            if capturing:
                sql_lines.append(line)
                if line.strip().endswith(';'):
                    break
        if sql_lines:
            return '\n'.join(sql_lines).strip()

        # 策略4: 整体就是SQL
        stripped = text.strip()
        if stripped.upper().startswith(('SELECT', 'WITH')):
            return clean_sql(stripped)

        return None

    def _finalize_success(self, exec_result, state_record: Dict) -> Dict:
        """构建成功的最终响应"""
        self.execution_result = {
            "success": True,
            "rows": exec_result.rows,
            "columns": exec_result.columns,
            "row_count": exec_result.row_count,
            "execution_time": exec_result.execution_time
        }
        self.state_history.append(state_record)

        return self._build_response(
            success=True,
            sql=self.generated_sql,
            result=self.execution_result
        )

    def _build_response(
        self,
        success: bool,
        sql: Optional[str] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """构建标准响应字典"""
        return {
            "success": success,
            "sql": sql,
            "result": result,
            "error": error,
            "trace": self.state_history,
            "correction_history": self.correction_history
        }


__all__ = ["ReactAgent", "AgentState", "CorrectionReviewResult"]
