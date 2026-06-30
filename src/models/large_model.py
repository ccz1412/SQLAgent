"""
大模型（智谱 AI API）调用模块

功能：
1. 封装智谱 AI API 调用（glm-4-flash）
2. 提供与 SmallModel 统一的接口
3. 支持流式输出（可选）
4. 错误处理和重试

使用示例：
    from src.models.large_model import LargeModel
    
    model = LargeModel()
    response = model.generate(
        messages=[{"role": "user", "content": "生成 SQL"}],
        max_tokens=1024
    )
    print(response)
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Generator
import time

# 从项目配置加载 API 配置
from src.utils.config_loader import load_config


class LargeModel:
    """
    大模型封装类（智谱 AI API）
    
    职责：
    1. 读取 API 配置（config/api_config.yaml）
    2. 调用智谱 AI API（glm-4-flash）
    3. 提供统一的 generate() 接口
    4. 错误处理和重试
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化大模型客户端
        
        参数：
            config_path: 配置文件路径（默认：config/api_config.yaml）
        """
        # 加载配置
        if config_path is None:
            # 自动查找配置文件（从当前文件向上查找）
            current = Path(__file__).parent.parent.parent
            config_path = current / "config" / "api_config.yaml"
        
        config = load_config(str(config_path))
        api_config = config.get("api", {})
        
        self.base_url = api_config.get("base_url", "https://open.bigmodel.cn/api/paas/v4")
        self.model_name = api_config.get("model_name", "glm-4-flash")
        self.api_key = api_config.get("api_key", "")
        self.max_retries = api_config.get("max_retries", 3)
        self.timeout = api_config.get("timeout", 60)
        
        # 初始化 OpenAI 客户端
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout
            )
            print(f"[LargeModel] API 客户端初始化成功 | 模型：{self.model_name}")
        except ImportError:
            raise ImportError("缺少依赖：openai。请运行：pip install openai")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stream: bool = False
    ) -> str:
        """
        生成文本（非流式）
        
        参数：
            messages: 消息列表，格式：[{
            max_tokens: 最大生成 token 数
            temperature: 温度
            top_p: 核采样参数
            stream: 是否流式输出（当前未实现）
        
        返回：
            生成的文本（str）
        """
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False
                )
                
                # 提取回复
                content = response.choices[0].message.content
                return content.strip() if content else ""
            
            except Exception as e:
                retries += 1
                print(f"[LargeModel] API 调用失败（重试 {retries}/{self.max_retries}）：{e}")
                if retries >= self.max_retries:
                    raise e
                time.sleep(2 ** retries)  # 指数退避
        
        return ""
    
    def generate_sql(
        self,
        question: str,
        schema: str,
        db_id: str,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> str:
        """
        生成 SQL（专用接口）
        
        参数：
            question: 用户问题
            schema: 数据库 schema
            db_id: 数据库 ID
            few_shot_examples: Few-shot 示例（可选）
        
        返回：
            生成的 SQL（str）
        """
        # 构造系统提示
        system_prompt = """你是一个 SQL 生成专家。根据用户的自然语言问题，生成对应的 SQL 查询语句。

规则：
1. 只输出 SQL 语句，不要有任何解释或前缀
2. 使用标准的 SQL 语法
3. 确保 SQL 可以在 SQLite 中执行
4. 如果问题不明确，生成最合理的 SQL
"""
        
        # 构造用户提示
        user_prompt = f"""数据库 Schema：
{schema}

问题：{question}

请生成 SQL 查询语句："""
        
        # 构造消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 添加 Few-shot 示例（如果有）
        if few_shot_examples:
            for example in few_shot_examples:
                messages.append({"role": "user", "content": example["question"]})
                messages.append({"role": "assistant", "content": example["sql"]})
        
        messages.append({"role": "user", "content": user_prompt})
        
        # 调用 API
        sql = self.generate(messages, max_tokens=512, temperature=0.1)
        
        # 清理 SQL（去除 markdown 代码块标记）
        sql = self._clean_sql(sql)
        
        return sql
    
    def correct_sql(
        self,
        wrong_sql: str,
        error_message: str,
        schema: str,
        question: str
    ) -> str:
        """
        纠正 SQL（专用接口）
        
        参数：
            wrong_sql: 错误的 SQL
            error_message: 错误信息
            schema: 数据库 schema
            question: 原始问题
        
        返回：
            纠正后的 SQL（str）
        """
        system_prompt = """你是一个 SQL 纠错专家。根据错误信息，修正 SQL 查询语句。

规则：
1. 只输出修正后的 SQL 语句
2. 仔细分析错误信息
3. 确保修正后的 SQL 可以正确执行
"""
        
        user_prompt = f"""数据库 Schema：
{schema}

原始问题：{question}

错误的 SQL：
{wrong_sql}

错误信息：
{error_message}

请修正 SQL："""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        sql = self.generate(messages, max_tokens=512, temperature=0.1)
        sql = self._clean_sql(sql)
        
        return sql
    
    def _clean_sql(self, sql: str) -> str:
        """清理 SQL（去除 markdown 标记等）"""
        # 去除 ```sql ... ``` 代码块
        if "```" in sql:
            # 提取代码块内容
            lines = sql.split("\n")
            in_code_block = False
            code_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    code_lines.append(line)
            sql = "\n".join(code_lines)
        
        # 去除 SQL 前的 "SQL:" 等前缀
        prefixes = ["SQL:", "sql:", "```sql", "以下是", "修正后的"]
        for prefix in prefixes:
            if sql.startswith(prefix):
                sql = sql[len(prefix):].strip()
        
        return sql.strip()
