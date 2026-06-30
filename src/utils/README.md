# utils/ 目录

## 功能描述
提供项目通用的工具函数，包括：
- 日志配置
- 配置文件加载
- SQL 处理辅助函数
- 文本格式化函数

## 文件列表

| 文件名 | 功能 | 主要接口 |
|--------|------|----------|
| `logger.py` | 日志配置 | `setup_logger()`, `get_logger()` |
| `config_loader.py` | 配置文件加载 | `load_config()`, `save_config()` |
| `helpers.py` | 辅助函数 | `format_prompt()`, `parse_sql_clauses()`, `clean_sql()` |

## 使用方法

### 日志
```python
from src.utils.logger import setup_logger, get_logger

# 初始化日志（在项目入口调用一次）
setup_logger(log_dir="logs", log_level="INFO")

# 获取 logger
logger = get_logger(__name__)
logger.info("这是一条日志")
```

### 配置加载
```python
from src.utils.config_loader import load_config

# 加载单个配置文件
model_config = load_config("config/model_config.yaml")

# 加载所有配置（返回合并后的字典）
all_config = load_config()  # 自动加载 config/ 下所有 .yaml 文件
```

### 辅助函数
```python
from src.utils.helpers import format_prompt, clean_sql

# 格式化 Prompt（将模板中的变量替换）
prompt = format_prompt(
    template="生成 SQL: {question}\nSchema: {schema}",
    question="列出所有学生",
    schema="CREATE TABLE student (...)"
)

# 清理 SQL（去除多余空格、注释等）
sql = clean_sql("  SELECT  * FROM  student  WHERE  age > 25  ")
```

## 运行依赖
- Python 3.10+
- PyYAML（`pip install pyyaml`）
- loguru（可选，`pip install loguru`）

## 注意事项
- 所有路径使用相对路径（相对于项目根目录）
- 配置文件格式为 YAML
- 日志默认保存在 `logs/` 目录
