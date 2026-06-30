#!/usr/bin/env python3
"""
Qwen3-Coder-30B 应答速度基准测试
导入 qwen_client 连接模块，测试：
  1. 首Token延迟（TTFT）
  2. 端到端延迟（E2E）
  3. 生成吞吐量（tokens/s）
  4. 不同长度 prompt 的影响
  5. 不同生成长度的影响
"""
import time
import statistics
import logging
import sys

# 导入连接模块
from qwen_client import chat, stream_chat, MODEL_NAME, BASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ============ 测试用例 ============

TEST_CASES = [
    {
        "name": "短问答-中文",
        "messages": [{"role": "user", "content": "什么是机器学习？用一句话回答"}],
        "warmup": True,
    },
    {
        "name": "短问答-英文",
        "messages": [{"role": "user", "content": "What is Python? Answer in one sentence."}],
    },
    {
        "name": "代码生成-Python",
        "messages": [{"role": "user", "content": "写一个Python函数，用二分查找在有序数组中查找元素"}],
        "max_tokens": 300,
    },
    {
        "name": "代码生成-快速排序",
        "messages": [{"role": "user", "content": "写一个完整的Python快速排序实现，包含注释"}],
        "max_tokens": 300,
    },
    {
        "name": "SQL查询-简单",
        "messages": [{"role": "user", "content": "写一条SQL语句，查询每个部门的平均工资，按工资降序排列"}],
        "max_tokens": 200,
    },
    {
        "name": "SQL查询-复杂",
        "messages": [
            {
                "role": "user",
                "content": (
                    "有以下三张表：users(id, name, dept_id)，orders(id, user_id, amount, created_at)，"
                    "departments(id, name)。写一条SQL查询2024年消费总额排名前10的部门。"
                ),
            }
        ],
        "max_tokens": 300,
    },
    {
        "name": "长文推理-分析",
        "messages": [
            {
                "role": "user",
                "content": (
                    "请详细解析快速排序算法的原理、时间复杂度分析、空间复杂度分析、"
                    "最坏情况与优化策略。用中文回答，确保完整。"
                ),
            }
        ],
        "max_tokens": 500,
    },
    {
        "name": "多轮对话",
        "messages": [
            {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "你好小明，有什么可以帮助你的？"},
            {"role": "user", "content": "我刚才说我叫什么？"},
        ],
        "max_tokens": 100,
    },
]

WARMUP_PROMPT = "hi"


def print_separator(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def format_time(seconds):
    if seconds is None:
        return "N/A"
    if seconds < 0.1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def test_sync_latency(case, round_num=1):
    """非流式：端到端延迟 + 吞吐量"""
    t0 = time.perf_counter()
    content, info = chat(
        case["messages"],
        temperature=0.0,
        max_tokens=case.get("max_tokens"),
        stream=False,
    )
    # info 已经是实际耗时，不再用 time.perf_counter() - t0
    return info


def test_stream_latency(case):
    """流式：首Token延迟 + 吞吐量"""
    gen, timing = stream_chat(
        case["messages"],
        temperature=0.0,
        max_tokens=case.get("max_tokens"),
    )

    # 消费流（不打印）
    for _ in gen:
        pass

    total = timing.get("total_s", 0)
    first_token = timing.get("first_token_s")
    completion_tokens = timing.get("completion_tokens", 0)
    tps = completion_tokens / total if total > 0 else 0

    return {
        "total_s": total,
        "first_token_s": first_token,
        "completion_tokens": completion_tokens,
        "tokens_per_second": tps,
    }


def run_benchmark():
    print_separator("Qwen3-Coder-30B 应答速度基准测试")
    logger.info(f"服务地址: {BASE_URL}")
    logger.info(f"模型名称: {MODEL_NAME}")
    logger.info(f"测试用例数: {len(TEST_CASES)}")
    print(f"\n{'用例名称':<20} {'首Token':>10} {'总耗时':>10} {'输出Tokens':>12} {'吞吐量':>12}")
    print("-" * 70)

    total_ttfts = []
    total_e2e = []
    total_tps = []

    for i, case in enumerate(TEST_CASES):
        name = case["name"]
        warmup = case.pop("warmup", False)

        if warmup:
            logger.info(f"[预热] 忽略首次调用的冷启动延迟...")
            t0 = time.perf_counter()
            _ = chat([{"role": "user", "content": WARMUP_PROMPT}],
                     temperature=0.0, max_tokens=10)
            warmup_time = time.perf_counter() - t0
            logger.info(f"  预热耗时: {format_time(warmup_time)}")
            # 预热后再测一次相同的内容（非 warmup 内容，跳过当前 case）
            # 不跳过，把它当作正常用例但排除首token冷启动影响
            # 重新执行流式测试
            info = test_stream_latency(case)
        else:
            info = test_stream_latency(case)

        first = info.get("first_token_s")
        total = info.get("total_s", 0)
        tokens = info.get("completion_tokens", 0)
        tps = info.get("tokens_per_second", 0)

        if first is not None:
            total_ttfts.append(first)
        total_e2e.append(total)
        total_tps.append(tps)

        print(
            f"{name:<20} {format_time(first):>10} "
            f"{format_time(total):>10} {tokens:>12} "
            f"{tps:>10.1f} tok/s"
        )

        # 每个用例之间短暂间隔
        time.sleep(1)

    # ============ 汇总 ============
    print_separator("汇总统计")
    print(f"\n{'指标':<20} {'最小值':>10} {'平均值':>10} {'最大值':>10}")
    print("-" * 55)

    if total_ttfts:
        print(
            f"{'首Token延迟(TTFT)':<20} {format_time(min(total_ttfts)):>10} "
            f"{format_time(statistics.mean(total_ttfts)):>10} "
            f"{format_time(max(total_ttfts)):>10}"
        )

    print(
        f"{'端到端延迟(E2E)':<20} {format_time(min(total_e2e)):>10} "
        f"{format_time(statistics.mean(total_e2e)):>10} "
        f"{format_time(max(total_e2e)):>10}"
    )

    print(
        f"{'生成吞吐量':<20} {min(total_tps):>10.1f} "
        f"{statistics.mean(total_tps):>10.1f} "
        f"{max(total_tps):>10.1f}"
        f" {' tok/s':>10}"
    )

    # ============ 第二轮：多轮测试看稳定性 ============
    print_separator("稳定性测试（同用例重复3次）")
    stable_case = {
        "messages": [{"role": "user", "content": "用Python快速排序算法对一个列表排序，写出完整代码"}],
        "max_tokens": 200,
    }

    print(f"\n{'次数':<8} {'首Token':>10} {'总耗时':>10} {'输出Tokens':>12} {'吞吐量':>12}")
    print("-" * 58)

    stable_ttfts = []
    stable_tps = []

    for r in range(1, 4):
        info = test_stream_latency(stable_case)
        first = info.get("first_token_s")
        total = info.get("total_s", 0)
        tokens = info.get("completion_tokens", 0)
        tps = info.get("tokens_per_second", 0)

        if first is not None:
            stable_ttfts.append(first)
        stable_tps.append(tps)

        print(
            f"第{r}次   {format_time(first):>10} "
            f"{format_time(total):>10} {tokens:>12} "
            f"{tps:>10.1f} tok/s"
        )
        time.sleep(1)

    print(f"\n{'指标':<20} {'均值':>10} {'标准差':>10}")
    print("-" * 35)
    if stable_ttfts:
        print(
            f"{'首Token延迟':<20} {format_time(statistics.mean(stable_ttfts)):>10} "
            f"{format_time(statistics.stdev(stable_ttfts)) if len(stable_ttfts) > 1 else 'N/A':>10}"
        )
    print(
        f"{'生成吞吐量':<20} {statistics.mean(stable_tps):>10.1f} "
        f"{statistics.stdev(stable_tps) if len(stable_tps) > 1 else 0:>10.1f} tok/s"
    )

    print_separator("测试完成")


if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        logger.error(f"测试中断: {e}")
        sys.exit(1)
