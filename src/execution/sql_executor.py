"""
SQL 执行引擎
安全地执行 SQL 语句，并返回格式化结果
"""

import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.utils.config_loader import load_db_root


@dataclass
class ExecutionResult:
    """SQL 执行结果"""
    success: bool
    rows: List[Tuple]
    columns: List[str]
    row_count: int
    execution_time: float
    error: Optional[str] = None


class SQLExecutor:
    """SQL 执行器"""
    
    def __init__(self, db_path: str):
        """
        初始化执行器
        
        Args:
            db_path: 数据库文件路径（相对于项目根目录）
        """
        self.db_path = self._resolve_path(db_path)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
    
    @classmethod
    def from_db_id(cls, db_id: str, data_root: Optional[str] = None) -> "SQLExecutor":
        """
        从数据库 ID 创建执行器

        默认从 config/db_config.yaml 的 spider.db_root 读取数据库根目录，
        也可通过 data_root 参数显式覆盖。

        Args:
            db_id: 数据库 ID（如 "student"）
            data_root: 数据库根目录（相对于项目根目录），None 时读取配置

        Returns:
            SQLExecutor 实例
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        if data_root is None:
            data_root = load_db_root()
        db_dir = project_root / data_root / db_id

        # 查找 .sqlite 或 .db 文件
        for ext in [".sqlite", ".db"]:
            db_file = db_dir / f"{db_id}{ext}"
            if db_file.exists():
                return cls(str(db_file))

        raise FileNotFoundError(f"未找到数据库文件: {db_dir}")


    
    def execute(self, sql: str, timeout: int = 30, max_rows: int = 100) -> ExecutionResult:
        """
        执行 SQL 语句
        
        Args:
            sql: SQL 语句
            timeout: 执行超时（秒）
            max_rows: 最大返回行数
            
        Returns:
            ExecutionResult 对象
        """
        start_time = time.time()
        
        # 安全检查：仅允许 SELECT（可扩展为支持 WITH）
        sql_upper = sql.strip().upper()
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return ExecutionResult(
                success=False,
                rows=[],
                columns=[],
                row_count=0,
                execution_time=0.0,
                error="仅支持 SELECT 语句"
            )
        
        # 自动添加 LIMIT（如果未指定）
        if "LIMIT" not in sql_upper and max_rows > 0:
            sql = sql.rstrip().rstrip(";")
            sql = f"{sql} LIMIT {max_rows};"
        
        try:
            # 连接数据库
            conn = sqlite3.connect(str(self.db_path), timeout=timeout)
            conn.execute("PRAGMA query_only = TRUE")  # 仅允许查询
            
            # 执行 SQL
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            conn.close()
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                rows=rows,
                columns=columns,
                row_count=len(rows),
                execution_time=execution_time
            )
        
        except sqlite3.Error as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                rows=[],
                columns=[],
                row_count=0,
                execution_time=execution_time,
                error=str(e)
            )
    
    def get_schema(self) -> str:
        """
        获取数据库 Schema（用于 Prompt）
        
        Returns:
            Schema 字符串（CREATE TABLE 语句）
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            schema_parts = []
            for table_name, create_sql in tables:
                if create_sql:
                    schema_parts.append(create_sql + ";")
            
            return "\n".join(schema_parts)
        
        except sqlite3.Error as e:
            return f"-- 获取 Schema 失败: {e}"
    
    @staticmethod
    def _resolve_path(relative_path: str) -> Path:
        """
        将相对路径解析为绝对路径
        
        Args:
            relative_path: 相对于项目根目录的路径
            
        Returns:
            绝对路径 Path 对象
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / relative_path


# 导出
__all__ = ["SQLExecutor", "ExecutionResult"]
