#!/usr/bin/env python3
"""
Qwen3-Coder-30B 推理服务调用脚本
支持重试、超时、详细错误诊断
"""
import sys
import time
import logging
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

# ============ 配置 ============
BASE_URL = "https://member.aicloud.szu.edu.cn:11443/inf-app/a19870157774385152564737/v1"
API_KEY = "unused"
USER_AGENT_KEY = "7595ead63e8b42609f5dcf750621aa91"
MODEL_NAME = "qwen3-coder-30b"

MAX_RETRIES = 3
RETRY_DELAY = 10  # 秒
REQUEST_TIMEOUT = 120  # 秒（模型推理需要时间）

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_client():
    """创建 OpenAI 客户端"""
    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        default_headers={"X-User-Agent-Key": USER_AGENT_KEY},
        timeout=REQUEST_TIMEOUT,
        max_retries=0,  # 我们自己管理重试
    )


def chat_once(client, messages, stream=False, **kwargs):
    """单次调用（不含重试）"""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=stream,
        **kwargs,
    )

    if stream:
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
        return None
    else:
        return resp.choices[0].message.content


def chat_with_retry(messages, **kwargs):
    """带重试的调用，含详细错误信息"""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"第 {attempt}/{MAX_RETRIES} 次尝试...")
            client = create_client()
            result = chat_once(client, messages, **kwargs)
            logger.info("调用成功！")
            return result

        except APIConnectionError as e:
            last_error = e
            logger.error(
                f"连接失败！服务可能尚未就绪。\n"
                f"  检查: 平台状态是否为「运行中」且算力消耗 > 0？\n"
                f"  详情: {e}"
            )
        except APITimeoutError as e:
            last_error = e
            logger.error(f"请求超时（{REQUEST_TIMEOUT}s），模型首次推理可能较慢")
        except APIError as e:
            last_error = e
            logger.error(f"API 错误: HTTP {e.status_code} - {e.message}")

            # 400 类错误不重试
            if e.status_code and 400 <= e.status_code < 500:
                logger.error("客户端错误，停止重试。请检查请求参数。")
                raise
        except Exception as e:
            last_error = e
            logger.error(f"未知错误: {type(e).__name__}: {e}")

        if attempt < MAX_RETRIES:
            logger.info(f"等待 {RETRY_DELAY}s 后重试...")
            time.sleep(RETRY_DELAY)

    # 全部失败，输出诊断建议
    logger.error("\n" + "=" * 60)
    logger.error("全部重试失败！诊断建议：")
    logger.error("1. 确认平台状态为「运行中」且 GPU 算力消耗 > 0")
    logger.error("2. 若算力为 0，说明 vLLM 进程崩溃，检查部署日志")
    logger.error("3. 确认 start.sh 已正确更新（PYTHONPATH 需包含模型目录）")
    logger.error(f"4. 手动测试: curl -k {BASE_URL}/models")
    logger.error("=" * 60)
    raise last_error


# ============ 测试用例 ============

def test_hello():
    """基础测试：问候"""
    logger.info("=" * 50)
    logger.info("测试 1: 基础问答")
    result = chat_with_retry([
        {"role": "user", "content": "你好，用一句话介绍你自己"}
    ])
    if result:
        print(f"\n回复:\n{result}\n")


def test_code():
    """代码生成测试"""
    logger.info("=" * 50)
    logger.info("测试 2: Python 快速排序")
    result = chat_with_retry([
        {"role": "user", "content": "写一个Python快速排序函数"}
    ])
    if result:
        print(f"\n回复:\n{result}\n")


def test_stream():
    """流式输出测试"""
    logger.info("=" * 50)
    logger.info("测试 3: 流式输出")
    chat_with_retry(
        [{"role": "user", "content": "数到10，逐一输出"}],
        stream=True,
    )


# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         # 交互模式：python call_qwen.py "你的问题"
#         prompt = sys.argv[1]
#         logger.info(f"单次问答: {prompt}")
#         result = chat_with_retry([{"role": "user", "content": prompt}])
#         if result:
#             print(result)
#     else:
#         # 默认运行全部测试
#         try:
#             test_hello()
#             test_code()
#             test_stream()
#         except Exception:
#             logger.error("\n请检查：")
#             logger.error("1. start.sh 是否已更新：PYTHONPATH 需包含 MODEL_PATH")
#             logger.error("2. 上传新的 deploy/start.sh 后重新部署")
#             sys.exit(1)
