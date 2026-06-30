#!/usr/bin/env python3
"""
API 连接模块（支持智谱 AI、Qwen 等 OpenAI 兼容 API）
提供客户端创建、单次调用、流式调用的封装
自动从 config/api_config.yaml 读取配置
"""
import time
import sys
import logging
from pathlib import Path
from openai import OpenAI

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger("api_client")


def _load_api_config():
    """从 config/api_config.yaml 加载 API 配置"""
    from src.utils.config_loader import load_config
    config = load_config()
    api_config = config.get("api", {})
    return {
        "base_url": api_config.get("base_url", ""),
        "api_key": api_config.get("api_key", ""),
        "model_name": api_config.get("model_name", ""),
        "user_agent_key": api_config.get("user_agent_key", ""),
        "timeout": api_config.get("timeout", 180),
    }


def get_client(timeout=None):
    """获取配置好的 OpenAI 客户端（自动读取配置）"""
    cfg = _load_api_config()
    base_url = cfg["base_url"]
    api_key = cfg["api_key"]
    user_agent_key = cfg["user_agent_key"]

    # 构建默认请求头
    default_headers = {}
    if user_agent_key:
        default_headers["X-User-Agent-Key"] = user_agent_key

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers if default_headers else None,
        timeout=timeout or cfg["timeout"],
        max_retries=0,
    )


def chat(messages, temperature=0.0, max_tokens=None, stream=False):
    """
    同步调用，返回 (content, 耗时信息字典)
    自动从 config/api_config.yaml 读取模型名称

    Returns:
        content: str — 回复内容
        timing: dict — {
            "total_s": 总耗时,
            "prompt_tokens": int,
            "completion_tokens": int,
            "tokens_per_second": float,
        }
    """
    t0 = time.perf_counter()
    cfg = _load_api_config()
    client = get_client()

    resp = client.chat.completions.create(
        model=cfg["model_name"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if stream:
        content = ""
        chunk_count = 0
        first_token_time = None
        for chunk in resp:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                content += chunk.choices[0].delta.content

        total_time = time.perf_counter() - t0
        return content, {
            "total_s": total_time,
            "first_token_s": first_token_time - t0 if first_token_time else None,
            "chunks": chunk_count,
            "completion_tokens": chunk_count,  # 近似
        }
    else:
        total_time = time.perf_counter() - t0
        choice = resp.choices[0]
        usage = resp.usage

        return choice.message.content, {
            "total_s": total_time,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "tokens_per_second": (
                usage.completion_tokens / total_time
                if usage and usage.completion_tokens and total_time > 0
                else 0
            ),
        }


def stream_chat(messages, **kwargs):
    """
    流式调用，yield 每个 token，完成后返回耗时信息。
    使用方式:
        gen, info = stream_chat(messages)
        for token in gen:
            print(token, end="")
        print(info)  # 结束时获取耗时
    """
    kwargs["stream"] = True
    result = []
    timing = {}

    t0 = time.perf_counter()
    first_token_time = None
    cfg = _load_api_config()
    client = get_client()

    resp = client.chat.completions.create(
        model=cfg["model_name"],
        messages=messages,
        **kwargs,
    )

    def generator():
        nonlocal first_token_time
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token = chunk.choices[0].delta.content
                result.append(token)
                yield token

        # 所有 token 输出完毕后设置 timing
        timing["total_s"] = time.perf_counter() - t0
        timing["first_token_s"] = (first_token_time - t0) if first_token_time else None
        timing["completion_tokens"] = len(result)

    gen = generator()
    return gen, timing
