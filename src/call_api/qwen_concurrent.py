#!/usr/bin/env python3
"""
基于 qwen_client 的并发批处理 API 调用工具
使用 ThreadPoolExecutor 并发请求 vLLM 服务
"""
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from qwen_client import get_client, MODEL_NAME

logger = logging.getLogger("qwen_concurrent")


def batch_chat(prompts, max_workers=8, timeout=180, temperature=0.0, max_tokens=None):
    """
    并发发送多个 prompt，返回结果列表（保持原始顺序）。

    Args:
        prompts: [(idx, messages), ...] 列表，idx 用于排序
        max_workers: 并发线程数
        timeout: 单次请求超时
        temperature: 采样温度
        max_tokens: 最大生成 token 数

    Returns:
        [(idx, content, timing_dict), ...] 列表，按 idx 排序
    """
    client = get_client(timeout=timeout)
    results = {}
    errors = []

    def _worker(idx, messages):
        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            content = resp.choices[0].message.content
            elapsed = time.perf_counter() - start
            usage = resp.usage
            return idx, content, {
                "success": True,
                "elapsed_s": elapsed,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            errors.append((idx, str(e)))
            return idx, None, {
                "success": False,
                "elapsed_s": elapsed,
                "error": str(e)[:200],
            }

    total = len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, idx, msgs): idx for idx, msgs in prompts}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            idx, content, info = future.result()
            results[idx] = (content, info)
            if info["success"]:
                logger.debug(
                    f"  [{completed}/{total}] idx={idx} OK "
                    f"({info['elapsed_s']:.1f}s, {info.get('completion_tokens', 0)} tok)"
                )
            else:
                logger.warning(f"  [{completed}/{total}] idx={idx} FAIL: {info['error'][:80]}")

    # 按 idx 排序返回
    sorted_results = [
        (idx, *results.get(idx, (None, {"success": False, "error": "missing"})))
        for idx in sorted([p[0] for p in prompts])
    ]

    if errors:
        logger.warning(f"  {len(errors)}/{total} requests failed, will retry")

    return sorted_results


def batch_chat_with_retry(prompts, max_workers=8, max_retries=3, **kwargs):
    """
    并发发送，失败项自动重试（最多 max_retries 次）。

    Returns:
        [(idx, content, timing_dict), ...] 列表
    """
    remaining = list(prompts)
    final_results = {}

    for attempt in range(1, max_retries + 1):
        if not remaining:
            break

        if attempt > 1:
            logger.info(f"  重试第 {attempt} 轮: {len(remaining)} 个请求...")
            time.sleep(2 * attempt)  # 递增延迟

        batch_results = batch_chat(remaining, max_workers=max_workers, **kwargs)

        remaining = []
        for idx, content, info in batch_results:
            if info["success"]:
                final_results[idx] = (content, info)
            else:
                remaining.append((idx, next((m for i, m in prompts if i == idx), [])))

    # 最终失败的
    for idx, _ in remaining:
        if idx not in final_results:
            final_results[idx] = (None, {"success": False, "error": "all retries exhausted"})

    # 按 idx 排序
    sorted_idx = sorted(final_results.keys())
    return [(idx, *final_results[idx]) for idx in sorted_idx]


def estimate_best_workers(max_concurrent=22):
    """
    根据 vLLM 的 max concurrency 估算最佳并发数。
    一般设为 max_concurrent 的 1/3 ~ 1/2，避免抢占过激。
    """
    return max(1, min(max_concurrent // 3, 16))
