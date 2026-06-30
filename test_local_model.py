"""
测试脚本：验证本地小模型（Llama-3.1-8B + LoRA）加载和推理

功能：
1. 测试模型加载（支持 4-bit 量化）
2. 测试 SQL 生成
3. 测试 SQL 纠错
4. 测试 ReAct Agent（使用本地小模型）

使用方法：
    # 测试本地小模型
    python test_local_model.py --use_small_model
    
    # 对比测试（本地小模型 vs API 大模型）
    python test_local_model.py --compare
    
    # 指定数据库
    python test_local_model.py --use_small_model --db_id student_db
"""

import argparse
import sys
import time
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def test_model_loading(use_small_model: bool = True):
    """
    测试模型加载
    
    Args:
        use_small_model: 是否测试本地小模型
    """
    print("=" * 70)
    print("测试 1：模型加载")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        if use_small_model:
            print(f"\n正在加载本地小模型...")
            print(f"  基础模型：dep/model/Meta-Llama-3___1-8B-Instruct")
            print(f"  LoRA 权重：exp/outputs/sql2sr_lora")
            print(f"  量化：4-bit (bitsandbytes)\n")
            
            from src.models.model_loader import get_model, ModelType
            model = get_model(ModelType.SMALL)
            
            load_time = time.time() - start_time
            print(f"✅ 模型加载成功！耗时：{load_time:.2f} 秒")
            print(f"  模型类型：{type(model).__name__}")
            
            return model
        
        else:
            print(f"\n正在初始化 API 大模型客户端...")
            print(f"  模型：glm-4-flash (智谱 AI)")
            print(f"  API 端点：https://open.bigmodel.cn/api/paas/v4\n")
            
            from src.models.model_loader import get_model, ModelType
            model = get_model(ModelType.LARGE)
            
            load_time = time.time() - start_time
            print(f"✅ 客户端初始化成功！耗时：{load_time:.2f} 秒")
            
            return model
    
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def test_sql_generation(model, use_small_model: bool, question: str, schema: str):
    """
    测试 SQL 生成
    
    Args:
        model: 模型实例
        use_small_model: 是否使用本地小模型
        question: 测试问题
        schema: 数据库 schema
    """
    print("\n" + "=" * 70)
    print("测试 2：SQL 生成")
    print("=" * 70)
    
    print(f"\n测试问题：{question}")
    print(f"数据库 Schema：\n{schema}\n")
    
    start_time = time.time()
    
    try:
        if use_small_model:
            # 使用本地小模型生成 SQL
            print(f"正在使用本地小模型生成 SQL...\n")
            
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是一个 SQL 生成专家。根据用户的自然语言问题，生成对应的 SQL 查询语句。

规则：
1. 只输出 SQL 语句，不要有任何解释或前缀
2. 使用标准的 SQL 语法
3. 确保 SQL 可以在 SQLite 中执行

<|eot_id|><|start_header_id|>user<|end_header_id|>

{schema}

问题：{question}

请生成 SQL：<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            
            sql = model.generate(prompt, max_tokens=512, temperature=0.1)
        
        else:
            # 使用 API 大模型生成 SQL
            print(f"正在调用智谱 AI API 生成 SQL...\n")
            
            sql = model.generate_sql(
                question=question,
                schema=schema,
                db_id="test"
            )
        
        generation_time = time.time() - start_time
        
        print(f"✅ SQL 生成成功！耗时：{generation_time:.2f} 秒")
        print(f"\n生成的 SQL：")
        print("-" * 70)
        print(sql)
        print("-" * 70)
        
        return sql
    
    except Exception as e:
        print(f"❌ SQL 生成失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def test_sql_execution(sql: str, db_id: str):
    """
    测试 SQL 执行
    
    Args:
        sql: 生成的 SQL
        db_id: 数据库 ID
    """
    print("\n" + "=" * 70)
    print("测试 3：SQL 执行")
    print("=" * 70)
    
    if not sql:
        print("\n⚠️ 跳过测试（SQL 为空）")
        return None
    
    print(f"\n正在执行 SQL...\n")
    print(f"SQL：{sql}\n")
    
    try:
        from src.execution.sql_executor import SQLExecutor
        
        executor = SQLExecutor.from_db_id(db_id)
        result = executor.execute(sql)
        
        if result.success:
            print(f"✅ SQL 执行成功！")
            print(f"  返回行数：{result.row_count}")
            print(f"  执行时间：{result.execution_time:.3f} 秒")
            
            if result.rows:
                print(f"\n前 5 行结果：")
                print("-" * 70)
                for i, row in enumerate(result.rows[:5]):
                    print(row)
                if result.row_count > 5:
                    print(f"... (共 {result.row_count} 行)")
                print("-" * 70)
            
            return result
        else:
            print(f"❌ SQL 执行失败：{result.error}")
            return None
    
    except Exception as e:
        print(f"❌ SQL 执行失败：{e}")
        return None


def test_react_agent(use_small_model: bool, question: str, db_id: str):
    """
    测试 ReAct Agent
    
    Args:
        use_small_model: 是否使用本地小模型
        question: 测试问题
        db_id: 数据库 ID
    """
    print("\n" + "=" * 70)
    print("测试 4：ReAct Agent")
    print("=" * 70)
    
    model_type = "本地小模型 (Llama-3.1-8B + LoRA)" if use_small_model else "远程大模型 (智谱 AI API)"
    print(f"\n使用模型：{model_type}")
    print(f"测试问题：{question}")
    print(f"数据库：{db_id}\n")
    
    try:
        from src.agent.react_agent_v2 import ReactAgentV2
        
        agent = ReactAgentV2(
            db_id=db_id,
            use_small_model=use_small_model,
            max_iterations=5,
            temperature=0.1
        )
        
        print(f"正在运行 ReAct Agent...\n")
        start_time = time.time()
        
        result = agent.run(question)
        
        total_time = time.time() - start_time
        
        print(f"\n{'=' * 70}")
        print("ReAct Agent 执行结果")
        print(f"{'=' * 70}")
        print(f"成功：{result['success']}")
        print(f"总耗时：{total_time:.2f} 秒")
        print(f"推理轮数：{len(result['trace'])}")
        
        if result['success']:
            print(f"\n最终 SQL：")
            print("-" * 70)
            print(result['sql'])
            print("-" * 70)
            
            if result.get('result'):
                print(f"\n执行结果（{result['result']['row_count']} 行）：")
                for i, row in enumerate(result['result']['rows'][:10]):
                    print(row)
                if result['result']['row_count'] > 10:
                    print(f"... (共 {result['result']['row_count']} 行)")
        
        else:
            print(f"\n错误：{result.get('error', '未知错误')}")
        
        print(f"\n推理轨迹：")
        for step in result['trace']:
            print(f"  第 {step['iteration']} 轮：{step['action']} → {step.get('result', '')[:80]}...")
        
        return result
    
    except Exception as e:
        print(f"❌ ReAct Agent 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(question: str, schema: str, db_id: str):
    """
    对比测试：本地小模型 vs API 大模型
    
    Args:
        question: 测试问题
        schema: 数据库 schema
        db_id: 数据库 ID
    """
    print("\n" + "=" * 70)
    print("对比测试：本地小模型 vs API 大模型")
    print("=" * 70)
    
    results = {}
    
    # 测试本地小模型
    print(f"\n{'-' * 70}")
    print("1. 本地小模型（Llama-3.1-8B + LoRA）")
    print(f"{'-' * 70}")
    
    try:
        from src.models.model_loader import get_model, ModelType
        small_model = get_model(ModelType.SMALL)
        
        sql_small = test_sql_generation(small_model, True, question, schema)
        result_small = test_sql_execution(sql_small, db_id)
        
        results['small_model'] = {
            'sql': sql_small,
            'execution_success': result_small.success if result_small else False,
            'row_count': result_small.row_count if result_small else 0
        }
    except Exception as e:
        print(f"本地小模型测试失败：{e}")
        results['small_model'] = {'error': str(e)}
    
    # 测试 API 大模型
    print(f"\n{'-' * 70}")
    print("2. API 大模型（智谱 AI glm-4-flash）")
    print(f"{'-' * 70}")
    
    try:
        from src.models.model_loader import get_model, ModelType
        large_model = get_model(ModelType.LARGE)
        
        sql_large = test_sql_generation(large_model, False, question, schema)
        result_large = test_sql_execution(sql_large, db_id)
        
        results['large_model'] = {
            'sql': sql_large,
            'execution_success': result_large.success if result_large else False,
            'row_count': result_large.row_count if result_large else 0
        }
    except Exception as e:
        print(f"API 大模型测试失败：{e}")
        results['large_model'] = {'error': str(e)}
    
    # 对比结果
    print(f"\n{'=' * 70}")
    print("对比结果")
    print(f"{'=' * 70}")
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        if 'error' in result:
            print(f"  错误：{result['error']}")
        else:
            print(f"  SQL：{result['sql'][:100]}...")
            print(f"  执行成功：{result['execution_success']}")
            print(f"  返回行数：{result['row_count']}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="测试本地小模型（Llama-3.1-8B + LoRA）加载和推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 测试本地小模型
  python test_local_model.py --use_small_model
  
  # 对比测试（本地小模型 vs API 大模型）
  python test_local_model.py --compare
  
  # 指定数据库和问题
  python test_local_model.py --use_small_model --db_id student_db --question "列出所有学生"
        """
    )
    
    parser.add_argument(
        "--use_small_model",
        action="store_true",
        help="测试本地小模型（默认：测试 API 大模型）"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比测试（本地小模型 vs API 大模型）"
    )
    parser.add_argument(
        "--db_id",
        type=str,
        default="student_db",
        help="数据库 ID（默认：student_db）"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="列出所有计算机科学专业的学生",
        help="测试问题"
    )
    parser.add_argument(
        "--test_all",
        action="store_true",
        help="运行所有测试（模型加载、SQL 生成、ReAct Agent）"
    )
    
    args = parser.parse_args()
    
    # 测试数据库 Schema（示例）
    schema = """Table: students (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    major TEXT,
    gpa REAL
)"""
    
    if args.compare:
        # 对比测试
        compare_models(args.question, schema, args.db_id)
    
    elif args.use_small_model:
        # 测试本地小模型
        print("\n" + "=" * 70)
        print("本地小模型测试")
        print("=" * 70)
        
        # 测试 1：模型加载
        model = test_model_loading(use_small_model=True)
        
        if model is None:
            print("\n❌ 模型加载失败，终止测试")
            sys.exit(1)
        
        # 测试 2：SQL 生成
        sql = test_sql_generation(model, use_small_model=True, question=args.question, schema=schema)
        
        # 测试 3：SQL 执行
        if sql:
            test_sql_execution(sql, args.db_id)
        
        # 测试 4：ReAct Agent
        if args.test_all:
            test_react_agent(use_small_model=True, question=args.question, db_id=args.db_id)
    
    else:
        # 测试 API 大模型
        print("\n" + "=" * 70)
        print("API 大模型测试")
        print("=" * 70)
        
        # 测试 1：客户端初始化
        model = test_model_loading(use_small_model=False)
        
        if model is None:
            print("\n❌ 客户端初始化失败，终止测试")
            sys.exit(1)
        
        # 测试 2：SQL 生成
        sql = test_sql_generation(model, use_small_model=False, question=args.question, schema=schema)
        
        # 测试 3：SQL 执行
        if sql:
            test_sql_execution(sql, args.db_id)
        
        # 测试 4：ReAct Agent
        if args.test_all:
            test_react_agent(use_small_model=False, question=args.question, db_id=args.db_id)
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
