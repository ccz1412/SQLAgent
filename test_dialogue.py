"""
Dialogue 模式自动化测试
模拟多轮对话：初始查询 → 追问过滤 → 追问排序
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dialogue.dialogue_manager import DialogueManager


def test_dialogue():
    print("=" * 60)
    print("Dialogue 模式自动化测试")
    print("=" * 60)

    # 初始化对话管理器
    dm = DialogueManager(db_id="student_db", use_small_model=False)

    # 测试场景
    test_cases = [
        ("列出所有学生", "Turn 1: 初始查询"),
        ("只看计算机科学专业的", "Turn 2: 追问过滤（含'只看'）"),
        ("按GPA从高到低排序", "Turn 3: 追问排序"),
        ("重置", "Turn 4: 重置对话"),
        ("列出软件工程专业的学生", "Turn 5: 新查询"),
    ]

    passed = 0
    failed = 0

    for message, description in test_cases:
        print(f"\n{'=' * 60}")
        print(f"[{description}]")
        print(f"用户: {message}")

        if message == "重置":
            dm.reset()
            print("对话已重置 ✓")
            passed += 1
            continue

        response = dm.process_message(message)

        print(f"Agent SQL: {response.get('sql', 'N/A')}")
        print(f"成功: {response.get('success', False)}")

        if response.get("success"):
            result = response.get("result", {})
            if result:
                print(f"返回 {result.get('row_count', 0)} 行:")
                for row in result.get("rows", [])[:5]:
                    print(f"  {row}")
            passed += 1
            print(f"  ✓ PASS")
        else:
            print(f"错误: {response.get('error', 'Unknown')}")
            failed += 1
            print(f"  ✗ FAIL")

        if response.get("is_follow_up"):
            print(f"  (检测为追问)")

    print(f"\n{'=' * 60}")
    print(f"结果: {passed} 通过, {failed} 失败 (共 {passed + failed})")
    print("=" * 60)

    # 显示对话历史
    print(f"\n对话历史 ({len(dm.get_history())} 轮):")
    for turn in dm.get_history():
        print(f"  Turn {turn['turn_id']}: {turn['user_message'][:30]}... → {turn['sql'][:40] if turn['sql'] else 'N/A'} ...")

    return failed == 0


if __name__ == "__main__":
    success = test_dialogue()
    sys.exit(0 if success else 1)
