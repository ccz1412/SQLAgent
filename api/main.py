"""
API 服务主入口
启动 FastAPI 应用
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from api import app
from src.utils.logger import setup_logger, get_logger


def main():
    """启动 API 服务"""
    # 初始化日志
    setup_logger(log_dir="logs", log_level="INFO")
    logger = get_logger(__name__)
    
    logger.info("正在启动 Multi-Turn Text-to-SQL Agent API 服务...")
    
    # 导入配置（可选）
    try:
        from src.utils.config_loader import load_config
        config = load_config("config/api_config.yaml")
        host = config.get("server", {}).get("host", "0.0.0.0")
        port = config.get("server", {}).get("port", 8000)
    except Exception:
        host = "0.0.0.0"
        port = 8000
    
    logger.info(f"服务将监听 {host}:{port}")
    
    # 启动 uvicorn
    import uvicorn
    uvicorn.run(
        app="api.main:app",
        host=host,
        port=port,
        reload=True,  # 开发模式：代码修改后自动重启
        log_level="info"
    )


if __name__ == "__main__":
    main()
