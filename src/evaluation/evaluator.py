"""
评估器模块

功能：
1. 加载数据集（Spider, BIRD, etc.）
2. 运行模型，生成预测 SQL
3. 计算评估指标
4. 生成评估报告

使用示例：
    from src.evaluation.evaluator import Evaluator
    
    evaluator = Evaluator(dataset="spider", split="dev")
    results = evaluator.evaluate(model_type="api")  # 使用 API 大模型
    evaluator.print_report()
    evaluator.save_results("results/spider_dev_api.json")
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.evaluation.metrics import exact_match, execution_accuracy, clause_accuracy

logger = get_logger(__name__)


class Evaluator:
    """
    评估器
    
    职责：
    1. 加载数据集
    2. 对每个样本运行模型（生成 SQL）
    3. 计算指标
    4. 生成报告
    """
    
    def __init__(self, dataset: str = "spider", split: str = "dev", db_dir: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            dataset: 数据集名称（"spider", "bird", "custom"）
            split: 数据分割（"train", "dev", "test"）
            db_dir: 数据库文件目录（默认：dat/spider_databases/）
        """
        self.dataset = dataset
        self.split = split
        self.db_dir = db_dir or (project_root / "dat" / "spider_databases")
        
        # 加载数据集
        self.data = self._load_dataset()
        logger.info(f"评估器初始化完成（dataset={dataset}, split={split}, size={len(self.data)}）")
    
    def _load_dataset(self) -> List[Dict]:
        """
        加载数据集
        
        Returns:
            数据样本列表，每个样本包含：
            - "question": 用户问题
            - "db_id": 数据库 ID
            - "query": 标准 SQL（Ground Truth）
        """
        logger.info(f"正在加载数据集：{self.dataset}/{self.split}")
        
        if self.dataset == "spider":
            # Spider 数据集格式
            data_path = project_root / "dat" / "spider" / f"{self.split}.json"
            
            if not data_path.exists():
                logger.error(f"数据集文件不存在：{data_path}")
                return []
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Spider 格式：列表 of {"question": ..., "db_id": ..., "query": ...}
            return data
        
        elif self.dataset == "bird":
            # BIRD 数据集格式
            data_path = project_root / "dat" / "bird" / f"{self.split}.json"
            
            if not data_path.exists():
                logger.error(f"数据集文件不存在：{data_path}")
                return []
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data
        
        else:
            logger.error(f"未知的数据集：{self.dataset}")
            return []
    
    def evaluate(
        self,
        model_type: str = "api",
        use_small_model: bool = False,
        max_samples: Optional[int] = None,
        verbose: bool = False
    ) -> Dict:
        """
        运行评估
        
        Args:
            model_type: 模型类型（"api", "small"）
            use_small_model: 是否使用本地小模型
            max_samples: 最大评估样本数（None = 全部）
            verbose: 是否打印详细信息
        
        Returns:
            评估结果字典
        """
        logger.info(f"开始评估（model_type={model_type}, use_small_model={use_small_model}）")
        
        # 初始化 Agent
        if model_type == "api":
            from src.agent.react_agent import ReactAgent
            agent = ReactAgent(db_dir=str(self.db_dir))
        else:
            from src.agent.react_agent_v2 import ReactAgentV2
            agent = ReactAgentV2(use_small_model=use_small_model)
        
        # 评估结果
        results = []
        metrics_summary = {
            "exact_match": 0,
            "execution_accuracy": 0,
            "total": 0
        }
        
        # 限制样本数
        data_to_eval = self.data[:max_samples] if max_samples else self.data
        
        # 逐样本评估
        for i, sample in enumerate(data_to_eval):
            question = sample["question"]
            db_id = sample["db_id"]
            gold_sql = sample["query"]
            
            if verbose:
                print(f"\n[{i+1}/{len(data_to_eval)}] 评估样本")
                print(f"  Question: {question}")
                print(f"  DB: {db_id}")
                print(f"  Gold SQL: {gold_sql}")
            
            # 运行 Agent，生成 SQL
            try:
                if model_type == "api":
                    result = agent.run(question, db_id)
                else:
                    result = agent.run(question, db_id)
                
                pred_sql = result.get("sql", "")
                pred_result = result.get("result")
                
                # 计算指标
                em = exact_match(pred_sql, gold_sql)
                
                # 执行准确率（需要执行 SQL）
                # 简化：这里只计算 Exact Match
                # 完整版需要连接数据库执行 SQL
                
                metrics_summary["exact_match"] += 1 if em else 0
                metrics_summary["total"] += 1
                
                # 记录结果
                results.append({
                    "sample_id": i,
                    "question": question,
                    "db_id": db_id,
                    "gold_sql": gold_sql,
                    "pred_sql": pred_sql,
                    "exact_match": em,
                    "success": em  # 简化：EM=True 则认为成功
                })
                
                if verbose:
                    print(f"  Pred SQL: {pred_sql}")
                    print(f"  Exact Match: {em}")
            
            except Exception as e:
                logger.error(f"评估样本 {i} 失败：{e}")
                results.append({
                    "sample_id": i,
                    "question": question,
                    "db_id": db_id,
                    "gold_sql": gold_sql,
                    "pred_sql": None,
                    "exact_match": False,
                    "success": False,
                    "error": str(e)
                })
        
        # 计算最终指标
        if metrics_summary["total"] > 0:
            metrics_summary["exact_match_rate"] = metrics_summary["exact_match"] / metrics_summary["total"]
        else:
            metrics_summary["exact_match_rate"] = 0.0
        
        logger.info(f"评估完成！Exact Match Rate: {metrics_summary['exact_match_rate']:.2%}")
        
        return {
            "metrics": metrics_summary,
            "results": results,
            "config": {
                "dataset": self.dataset,
                "split": self.split,
                "model_type": model_type,
                "use_small_model": use_small_model,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def print_report(self, eval_results: Optional[Dict] = None):
        """
        打印评估报告
        
        Args:
            eval_results: 评估结果（如果为 None，则需要先运行 evaluate()）
        """
        if eval_results is None:
            print("请先运行 evaluate()")
            return
        
        metrics = eval_results["metrics"]
        
        print("=" * 60)
        print("评估报告")
        print("=" * 60)
        print(f"\n数据集：{self.dataset}/{self.split}")
        print(f"样本数：{metrics['total']}")
        print(f"\n指标：")
        print(f"  Exact Match Rate: {metrics['exact_match']}/{metrics['total']} ({metrics['exact_match_rate']:.2%})")
        
        # 打印错误样本（前 5 个）
        errors = [r for r in eval_results["results"] if not r.get("success", False)]
        if errors:
            print(f"\n错误样本（前 5 个）：")
            for i, error in enumerate(errors[:5]):
                print(f"  {i+1}. Question: {error['question']}")
                print(f"     Gold SQL: {error['gold_sql']}")
                print(f"     Pred SQL: {error.get('pred_sql', 'N/A')}")
                if "error" in error:
                    print(f"     Error: {error['error']}")
        print()
    
    def save_results(self, eval_results: Dict, output_path: Optional[str] = None):
        """
        保存评估结果到 JSON 文件
        
        Args:
            eval_results: 评估结果
            output_path: 输出路径（默认：results/{dataset}_{split}_{timestamp}.json）
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = project_root / "results" / f"{self.dataset}_{self.split}_{timestamp}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存 to {output_path}")


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("评估器测试")
    print("=" * 60)
    
    # 测试 1：初始化
    print("\n[测试 1] 初始化评估器")
    try:
        evaluator = Evaluator(dataset="spider", split="dev")
        print(f"  ✅ 评估器初始化成功（数据集大小：{len(evaluator.data)}）")
    except Exception as e:
        print(f"  ❌ 初始化失败：{e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("\n提示：完整评估需要数据集文件（dat/spider/dev.json）")
