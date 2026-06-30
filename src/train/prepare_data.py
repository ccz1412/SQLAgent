"""
数据准备脚本：将 brid_trajectory 和 spider_trajectory 数据集
转换为 LLaMA-Factory 的 SFT 训练格式（instruction/output/system）
"""
import json
import os
import argparse

# ============ 提示词模板（与SHARE-main中BAM训练保持一致）============
SYSTEM_PROMPT = "You are an expert about text-to-SQL and pandas code."

SQL2SR_USER_PROMPT = """SR is a piece of pandas-like code, which is a intermediate representation between the natural language and SQL. I will provide you:
1. Schema: A python list and each element is a `table_name`.`column_name` string. It indicates that the table and column you could use in the SR.
2. SQL: The SQL that needed to be converted to SR
 
Your task is to generate valid SR which reflect the accurate logic in the SQL. Later, the SR will be converted to SQL.
Please pay attention that SR ignore 'join' action. Do not generate 'join' action.

schema = {schema}
sql = "{sql}"

Now generate the valid SR that display the reasoning process of generating SQL that can accurately answer the question:
```SR
[Your Answer]
```"""

ASSISTANT_PROMPT = """```SR
{sr}
```"""


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prepare_single_entry(item: dict) -> dict:
    """将单条数据转换为训练格式"""
    schema_str = item.get("schema_str", "")
    gold_sql = item.get("gold_sql", "").strip()
    trajectory = item.get("trajectory", "").strip()

    # 构建 instruction（用户输入）
    instruction = SQL2SR_USER_PROMPT.format(schema=schema_str, sql=gold_sql)

    # 构建 output（模型输出）
    output = ASSISTANT_PROMPT.format(sr=trajectory)

    return {
        "instruction": instruction,
        "output": output,
        "system": SYSTEM_PROMPT,
    }


def prepare_dataset(input_path: str) -> list:
    """加载并转换一个数据集"""
    print(f"  加载: {input_path}")
    data = load_json(input_path)
    result = []
    skipped = 0
    for item in data:
        # 跳过没有 trajectory 的数据
        trajectory = item.get("trajectory", "").strip()
        gold_sql = item.get("gold_sql", "").strip()
        if not trajectory or not gold_sql:
            skipped += 1
            continue
        entry = prepare_single_entry(item)
        result.append(entry)
    print(f"    有效条目: {len(result)}, 跳过: {skipped}")
    return result


def main():
    parser = argparse.ArgumentParser(description="准备 SQL2SR SFT 训练数据")
    parser.add_argument("--data_dir", type=str,
                        default="/home/user4/XiaZY/xia-zhenyu/dat",
                        help="数据集目录")
    parser.add_argument("--output_dir", type=str,
                        default="/home/user4/XiaZY/xia-zhenyu/exp/ft_data",
                        help="输出目录")
    parser.add_argument("--output_name", type=str,
                        default="sql2sr_train_data",
                        help="输出文件名（不含扩展名）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 合并 brid_trajectory 和 spider_trajectory
    all_data = []
    for subset in ["brid_trajectory", "spider_trajectory"]:
        json_path = os.path.join(args.data_dir, subset, "train_trajectory.json")
        if os.path.exists(json_path):
            print(f"处理数据集: {subset}")
            subset_data = prepare_dataset(json_path)
            all_data.extend(subset_data)
        else:
            print(f"警告: 未找到 {json_path}")

    print(f"\n总共 {len(all_data)} 条训练数据")

    # 保存训练数据
    output_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    save_json(all_data, output_path)
    print(f"训练数据已保存到: {output_path}")

    # 生成 dataset_info.json（LLaMA-Factory 格式）
    dataset_info = {
        args.output_name: {
            "file_name": f"{args.output_name}.json",
            "columns": {
                "prompt": "instruction",
                "response": "output",
                "system": "system",
            },
        }
    }
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    # 如果已有 dataset_info.json，则合并
    if os.path.exists(info_path):
        existing = load_json(info_path)
        existing.update(dataset_info)
        dataset_info = existing
    save_json(dataset_info, info_path)
    print(f"数据集配置已保存到: {info_path}")


if __name__ == "__main__":
    main()
