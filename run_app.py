"""
项目运行入口
支持多种运行模式：
- api: 启动 FastAPI 服务（默认）
- agent: 直接运行 ReAct Agent（命令行模式）
- dialogue: 运行多轮对话（交互式命令行）
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def run_api():
    """启动 FastAPI 服务"""
    print("正在启动 API 服务...")
    print("访问地址: <http://localhost:8000>")
    print("API 文档: <http://localhost:8000/docs>")
    print("按 Ctrl+C 停止服务\n")
    
    from api.main import main
    main()


def run_agent(question: str, db_id: str, use_small_model: bool = False):
    """
    直接运行 ReAct Agent（命令行模式）
    
    Args:
        question: 用户问题
        db_id: 数据库 ID
        use_small_model: 是否使用本地小模型
    """
    print(f"=== ReAct Agent 测试 ===")
    print(f"问题: {question}")
    print(f"数据库: {db_id}")
    print(f"模型: {'本地小模型 (Llama-3.1-8B + LoRA)' if use_small_model else '远程大模型 (智谱 AI API)'}\n")
    
    # 使用 V2 版本的 Agent（支持模型选择）
    from src.agent.react_agent_v2 import ReactAgentV2
    
    agent = ReactAgentV2(
        db_id=db_id,
        use_small_model=use_small_model,
        max_iterations=5,
        temperature=0.1
    )
    response = agent.run(question)
    
    print(f"\n=== 结果 ===")
    print(f"成功: {response['success']}")
    print(f"SQL: {response.get('sql', 'N/A')}")
    
    if response.get("result"):
        print(f"返回行数: {response['result']['row_count']}")
        print(f"执行时间: {response['result']['execution_time']:.3f}s")
    
    if response.get("error"):
        print(f"错误: {response['error']}")
    
    print(f"\n=== 推理轨迹 ===")
    for record in response.get("trace", []):
        print(f"第 {record['iteration']} 轮: {record['action']} → {record.get('result', '')}")


def run_dialogue(db_id: str, use_small_model: bool = False):
    """
    运行多轮对话（交互式命令行）
    
    Args:
        db_id: 数据库 ID
        use_small_model: 是否使用本地小模型
    """
    print(f"=== 多轮对话模式 ===")
    print(f"数据库: {db_id}")
    print(f"模型: {'本地小模型 (Llama-3.1-8B + LoRA)' if use_small_model else '远程大模型 (智谱 AI API)'}")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'reset' 重置对话\n")
    
    from src.dialogue.dialogue_manager import DialogueManager
    
    dm = DialogueManager(db_id=db_id, use_small_model=use_small_model)
    
    while True:
        try:
            user_input = input("用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("再见！")
            break
        
        if user_input.lower() in ["reset", "重置"]:
            dm.reset()
            print("对话已重置。\n")
            continue
        
        # 处理消息
        response = dm.process_message(user_input)
        
        print(f"\nAgent: ")
        print(f"  SQL: {response.get('sql', 'N/A')}")
        
        if response.get("result"):
            print(f"  返回: {response['result']['row_count']} 行")
        
        if response.get("error"):
            print(f"  错误: {response['error']}")
        
        print()


def run_evaluate(dataset: str = "spider", split: str = "dev", use_small_model: bool = False, max_samples: Optional[int] = None):
    """
    运行评估
    
    Args:
        dataset: 数据集名称（"spider", "bird", "custom"）
        split: 数据分割（"train", "dev", "test"）
        use_small_model: 是否使用本地小模型
        max_samples: 最大评估样本数（None = 全部）
    """
    print(f"=== 评估模式 ===")
    print(f"数据集: {dataset}/{split}")
    print(f"模型: {'本地小模型 (Llama-3.1-8B + LoRA)' if use_small_model else '远程大模型 (智谱 AI API)'}")
    if max_samples:
        print(f"样本数: {max_samples} (限制)")
    print()
    
    from src.evaluation.evaluator import Evaluator
    
    evaluator = Evaluator(dataset=dataset, split=split)
    
    results = evaluator.evaluate(
        model_type="small" if use_small_model else "api",
        use_small_model=use_small_model,
        max_samples=max_samples,
        verbose=True
    )
    
    evaluator.print_report(results)
    
    # 保存结果
    evaluator.save_results(results)
    
    print(f"\n评估完成！结果已保存")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="Multi-Turn Text-to-SQL Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 启动 API 服务
  python run_app.py api
  
  # 直接运行 Agent
  python run_app.py agent --question "列出所有学生" --db_id student_db
  
  # 使用本地小模型（Llama-3.1-8B + LoRA）
  python run_app.py agent -q "列出所有学生" -d student_db --use_small_model
  
  # 运行多轮对话
  python run_app.py dialogue --db_id student_db
  
  # 运行评估
  python run_app.py evaluate --dataset spider --split dev
  
  # 使用本地小模型运行评估（限制 10 个样本）
  python run_app.py evaluate -d spider -s dev --use_small_model --max_samples 10
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["api", "agent", "dialogue", "evaluate"],
        default="api",
        nargs="?",
        help="运行模式（默认: api）"
    )
    parser.add_argument("--question", "-q", help="用户问题（agent 模式）")
    parser.add_argument("--db_id", "-d", help="数据库 ID")
    parser.add_argument(
        "--use_small_model",
        action="store_true",
        help="是否使用本地小模型（Llama-3.1-8B + LoRA），默认使用 API 大模型"
    )
    parser.add_argument("--dataset", default="spider", help="数据集名称（evaluate 模式，默认: spider）")
    parser.add_argument("--split", default="dev", help="数据分割（evaluate 模式，默认: dev）")
    parser.add_argument("--max_samples", type=int, help="最大评估样本数（evaluate 模式，默认: 全部）")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        run_api()
    elif args.mode == "agent":
        if not args.question or not args.db_id:
            print("错误: agent 模式需要 --question 和 --db_id 参数")
            print("示例: python run_app.py agent -q '列出所有学生' -d student_db")
            sys.exit(1)
        run_agent(args.question, args.db_id, args.use_small_model)
    elif args.mode == "dialogue":
        if not args.db_id:
            print("错误: dialogue 模式需要 --db_id 参数")
            print("示例: python run_app.py dialogue -d student_db")
            sys.exit(1)
        run_dialogue(args.db_id, args.use_small_model)
    elif args.mode == "evaluate":
        run_evaluate(
            dataset=args.dataset,
            split=args.split,
            use_small_model=args.use_small_model,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
