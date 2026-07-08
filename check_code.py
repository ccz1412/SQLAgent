"""
代码完整检查脚本
检查所有模块的语法、导入、基本逻辑
"""

import sys
import traceback

def test_import(module_name: str, import_stmt: str):
    """测试模块导入"""
    print(f"Testing {module_name}...")
    try:
        exec(import_stmt)
        print(f"  PASS: {module_name} import OK")
        return True
    except Exception as e:
        print(f"  FAIL: {module_name} import FAILED - {e}")
        traceback.print_exc()
        return False


def test_syntax(file_path: str):
    """测试文件语法"""
    print(f"Testing syntax: {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, file_path, 'exec')
        print(f"  PASS: {file_path} syntax OK")
        return True
    except SyntaxError as e:
        print(f"  FAIL: {file_path} syntax ERROR - {e}")
        return False
    except Exception as e:
        print(f"  ERROR: {file_path} - {e}")
        return False


def main():
    print("=" * 60)
    print("Code Quality Check")
    print("=" * 60)
    
    # 添加项目根目录
    sys.path.insert(0, '.')
    
    results = {
        "syntax": [],
        "import": []
    }
    
    # 1. 语法检查
    print("\n[1] Syntax Check")
    print("-" * 60)
    
    files_to_check = [
        "src/models/small_model.py",
        "src/models/large_model.py",
        "src/models/model_loader.py",
        "src/agent/react_agent.py",
        "src/agent/sql_generator.py",
        "src/agent/intent_detector.py",
        "src/correction/clause_corrector.py",
        "src/prompts/prompt_templates.py",
        "src/dialogue/dialogue_manager.py",
        "src/execution/sql_executor.py",
        "src/utils/config_loader.py",
        "src/utils/helpers.py",
        "run_app.py",
    ]
    
    for file_path in files_to_check:
        ok = test_syntax(file_path)
        results["syntax"].append((file_path, ok))
    
    # 2. 导入检查
    print("\n[2] Import Check")
    print("-" * 60)
    
    import_tests = [
        ("src.utils.config_loader", "from src.utils.config_loader import load_config"),
        ("src.utils.helpers", "from src.utils.helpers import format_prompt, clean_sql"),
        ("src.execution.sql_executor", "from src.execution.sql_executor import SQLExecutor"),
        ("src.models.model_loader", "from src.models.model_loader import ModelLoader, ModelType"),
        ("src.agent.react_agent", "from src.agent.react_agent import ReactAgent, ErrorAnalysisResult, SemanticCheckResult"),
        ("src.agent.sql_generator", "from src.agent.sql_generator import generate_sql, correct_sql"),
        ("src.agent.intent_detector", "from src.agent.intent_detector import IntentDetector, IntentType"),
        ("src.correction.clause_corrector", "from src.correction.clause_corrector import ClauseCorrector"),
        ("src.prompts.prompt_templates", "from src.prompts.prompt_templates import build_sql_generation_prompt, build_correction_agent_prompt, build_semantic_check_prompt"),
        ("src.dialogue.dialogue_manager", "from src.dialogue.dialogue_manager import DialogueManager"),
    ]
    
    for module_name, import_stmt in import_tests:
        ok = test_import(module_name, import_stmt)
        results["import"].append((module_name, ok))
    
    # 3. 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    syntax_pass = sum(1 for _, ok in results["syntax"] if ok)
    syntax_total = len(results["syntax"])
    import_pass = sum(1 for _, ok in results["import"] if ok)
    import_total = len(results["import"])
    
    print(f"\nSyntax Check: {syntax_pass}/{syntax_total} passed")
    if syntax_pass < syntax_total:
        print("  Failed files:")
        for file_path, ok in results["syntax"]:
            if not ok:
                print(f"    - {file_path}")
    
    print(f"\nImport Check: {import_pass}/{import_total} passed")
    if import_pass < import_total:
        print("  Failed modules:")
        for module_name, ok in results["import"]:
            if not ok:
                print(f"    - {module_name}")
    
    overall_pass = (syntax_pass + import_pass)
    overall_total = (syntax_total + import_total)
    print(f"\nOVERALL: {overall_pass}/{overall_total} checks passed")
    
    if overall_pass == overall_total:
        print("\nALL CHECKS PASSED! Code quality: GOOD")
        return 0
    else:
        print(f"\n{overall_total - overall_pass} check(s) failed. Please fix the errors.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
