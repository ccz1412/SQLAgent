# execution/ 目录

## 功能描述
SQL 执行引擎，负责安全地执行生成的 SQL 语句，并返回格式化后的结果。

## 文件列表

| 文件名 | 功能 | 主要接口 |
|--------|------|----------|
| `__init__.py` | 模块初始化 | - |
| `sql_executor.py` | SQL 执行器 | `SQLExecutor`, `ExecutionResult` |

## 使用方法

### 基本使用
```python
from src.execution.sql_executor import SQLExecutor

# 初始化执行器（SQLite）
executor = SQLExecutor(db_path="data/test_databases/spider.db")

# 执行 SQL
result = executor.execute("SELECT * FROM student WHERE age > 25")

if result.success:
    print(f"返回 {result.row_count} 行")
    for row in result.rows:
        print(row)
else:
    print(f"执行失败: {result.error}")
```

### 与 Agent 集成
```python
# 在 ReAct Agent 的 Tools 中调用
def execute_sql(sql: str, db_id: str) -> dict:
    executor = SQLExecutor.from_db_id(db_id)
    result = executor.execute(sql)
    
    return {
        "success": result.success,
        "rows": result.rows,
        "columns": result.columns,
        "error": result.error,
        "execution_time": result.execution_time
    }
```

## 安全特性
- ✅ 仅允许 SELECT 语句（防止数据修改）
- ✅ 自动添加 LIMIT（防止返回过多数据）
- ✅ SQL 注入检测（简单规则）
- ✅ 执行超时控制

## 运行依赖
- Python 3.10+
- SQLite3（标准库）

## 注意事项
- 默认连接 SQLite 数据库
- 可以通过 `db_config.yaml` 配置其他数据库（MySQL, PostgreSQL）
- 执行结果默认最多返回 100 行（可在配置中修改）
