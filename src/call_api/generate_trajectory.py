#!/usr/bin/env python3
"""
[并发版] Spider 动作轨迹数据集生成脚本（SHARE SR 格式）
使用 Qwen API (vLLM 并发) 将 Spider train 的 SQL 转换为 SHARE 风格的 SR 轨迹

与原版区别：
- 使用 ThreadPoolExecutor 并发请求，大幅提速（8~12x）
- 批量处理 + 自动重试失败项
- 保留断点续传
"""
import json
import os
import sys
import time
import re
import logging
import argparse

# 导入并发批处理工具
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_concurrent import batch_chat_with_retry, estimate_best_workers

# ============ 配置 ============
SPIDER_DIR = "/home/user4/XiaZY/dataset/spider"
TRAIN_FILE = os.path.join(SPIDER_DIR, "train_spider.json")
TABLES_FILE = os.path.join(SPIDER_DIR, "tables.json")

OUTPUT_DIR = "/home/user4/XiaZY/dataset/spider_trajectory"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_trajectory.json")

BATCH_SIZE = 32     # 每批处理的样本数
MAX_WORKERS = 8     # 并发请求数（会被 estimate_best_workers 自适应调整）
MAX_RETRIES = 3     # 失败重试次数
REQUEST_TIMEOUT = 180
MAX_MODEL_LEN = 8192  # 与 start.sh 中的 MAX_MODEL_LEN 保持一致，改动后并发数自动适配

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "generate.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============ SHARE 风格的 Few-Shot 示例 ============
SPIDER_SQL2SR_SHOTS = """question = "How many movies directed by Francis Ford Coppola have a popularity of more than 1,000? Please also show the critic of these movies."
schema = [movies.movie_title, ratings.critic, movies.director_name, movies.movie_popularity, ratings.movie_id, movies.movie_id']
evidence = "Francis Ford Coppola refers to director_name; popularity of more than 1,000 refers to movie_popularity >1000" 
SQL = "SELECT T2.movie_title, T1.critic FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.director_name = 'Francis Ford Coppola' AND T2.movie_popularity > 1000"
```SR
df1 = df.where(element = movies.director_name, filter = 'Francis Ford Coppola')
df2 = df1.where(element = movies.movie_popularity, filter = '> 1000')
res = df2.select(movies.movie_title, ratings.critic)
```
    
question = "For all the movies which were produced by cruel and unusual films, which one has the most popularity?"
schema = [movie.title, `production company`.company_id, movie_company.company_id, movie_company.movie_id, movie.movie_id, `production company`.company_name, movie.popularity]
evidence = "produced by cruel and unusual films refers to company_name = 'Cruel and Unusual Films'; most popularity refers to max(popularity)"
SQL = "SELECT T3.title FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id WHERE T1.company_name = 'Cruel and Unusual Films' ORDER BY T3.popularity DESC LIMIT 1"
```SR
df1 = df.where(element = `production company`.company_name, filter = 'Cruel and Unusual Films')
df2 = df1.orderby(by = movie.popularity, desc).limit(1)
res = df2.select(movie.title)
```

question = "Among the professors who have more than 3 research assistants, how many of them are male?"
schema = [prof.gender, RA.student_id, RA.prof_id, prof.prof_id]
evidence = "research assistant refers to the student who serves for research where the abbreviation is RA; more than 3 research assistant refers to COUNT(student_id) > 3;"
SQL = "SELECT COUNT(*) FROM ( SELECT T2.prof_id FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id WHERE T2.gender = 'Male' GROUP BY T1.prof_id HAVING COUNT(T1.student_id) > 3 )"
```SR
df1 = df.groupby(prof.prof_id).having(element = count(RA.student_id), filter = '> 3')
df2 = df1.where(element = 'prof.gender', filter = 'Male')
res = df2.count()
```

question = "Find the id of courses which are registered or attended by student whose id is 121?"
schema = [student_course_registrations.course_id, student_course_registrations.student_id, student_course_attendance.course_id, student_course_attendance.student_id]
evidence = ""
SQL = "SELECT course_id FROM student_course_registrations WHERE student_id = 121 UNION SELECT course_id FROM student_course_attendance WHERE student_id = 121"
```SR
df1 = df.where(student_course_registrations.student_id, filter = 121)
tmp_res1 = df1.select(student_course_registrations.course_id)

df2 = df.where(student_course_attendance.student_id, filter = 121)
tmp_res2 = df2.select(student_course_attendance.course_id)

res = tmp_res1.union(tmp_res2)
```

question = "What are the names of all tracks that are on the movies playlist but not in the music playlist?"
schema = [tracks.name, tracks.id, playlist_tracks.track_id, playlist_tracks.playlist_id, playlists.id, playlists.name]
evidence = ""
SQL = "SELECT T1.name FROM tracks AS T1 JOIN playlist_tracks AS T2 ON T1.id  =  T2.track_id JOIN playlists AS T3 ON T2.playlist_id  =  T3.id WHERE T3.name  =  'Movies' EXCEPT SELECT T1.name FROM tracks AS T1 JOIN playlist_tracks AS T2 ON T1.id  =  T2.track_id JOIN playlists AS T3 ON T2.playlist_id  =  T3.id WHERE T3.name  =  'Music'"
```SR
df1 = df.where(playlists.name, filter = 'Movies')
tmp_res1 = df1.select(tracks.name)

df2 = df.where(playlists.name, filter = 'Music')
tmp_res2 = df2.select(tracks.name)

res = tmp_res1.except(tmp_res2)
```

question = "Which directors had a movie both in the year 1999 and 2000?"
schema = [movie.director, movie.Year]
evidence = ""
SQL = "SELECT director FROM movie WHERE YEAR  =  2000 INTERSECT SELECT director FROM movie WHERE YEAR  =  1999"
```SR
df1 = df.where(movie.YEAR, filter = 2000)
tmp_res1 = df1.select(movie.director)

df2 = df.where(movie.YEAR, filter = 1999)
tmp_res2 = df2.select(movie.director)

res = tmp_res1.intersect(tmp_res2)
```"""

SQL2SR_PROMPT = """"You are a text-to-SQL expert. SR is a piece of pandas-like code, which is a intermediate representation between the natural language and SQL. Given the database schema, question, evidence and SQL, your task is convert the SQL to SR which reflect the accurate logic in the SQL. 
I'll provide several example to you to help you understand the syntax of the SR and the convertion logic. SR ignore 'join' action. Do not generate 'join' action.

```Examples
{shots}
```

Now convert the following SQL to valid SR based on the database schema, question and evidence.
question = "{question}"
schema = "{schema}"
evidence = "{evidence}"
SQL = "{sql}"
```SR
[Your Answer]
```
"""


# ============ 工具函数（与原版相同）============

def get_column_table_spider(table_json, question_info, generated_sql):
    db_id = question_info['db_id']
    table_info = None
    for content in table_json:
        if content['db_id'] == db_id:
            table_info = content
            break
    if table_info is None:
        return []

    table_names_list = table_info["table_names_original"]
    column_names_list = [
        [table_names_list[int(content[0])], content[1]]
        for content in table_info['column_names_original'][1:]
    ]
    pure_column_name_list = [i[1] for i in column_names_list]

    filtered_tables = []
    filtered_columns = []
    for table in table_names_list:
        if table in generated_sql:
            filtered_tables.append(table)
    for column in pure_column_name_list:
        if column in generated_sql:
            filtered_columns.append(column)

    filtered_tables = list(set(filtered_tables))
    filtered_columns = list(set(filtered_columns))

    final_columns = []
    for columns in filtered_columns:
        tuples = _find_table(column_names_list, columns)
        for tup in tuples:
            if tup[0] in filtered_tables:
                final_columns.append(tup)

    if '*' in generated_sql:
        stars = [[t, '*'] for t in filtered_tables]
        final_columns = stars + final_columns

    return final_columns


def _find_table(table_column_list, column):
    res = []
    for table_column in table_column_list:
        if table_column[1] == column:
            res.append(table_column)
    return res


def format_schema_for_prompt(schema_list):
    formatted = []
    for table, column in schema_list:
        if ' ' in table or '-' in table:
            table = f"`{table}`"
        if ' ' in column or '-' in column:
            column = f"`{column}`"
        formatted.append(f"{table}.{column}")
    return str(formatted)


def extract_sr_from_response(response):
    if not response:
        return ""

    # 1) ```SR ... ```
    match = re.search(r"```SR\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2) ```SR 开头缺闭合
    match2 = re.search(r"```SR\s*(.*)", response, re.DOTALL | re.IGNORECASE)
    if match2:
        content = match2.group(1).strip()
        content = re.sub(r"```$", "", content).strip()
        return content

    # 3) 裸写 SR 代码（无 ```SR 标记）
    sr_keywords = r"\bdf\d*\s*\.\s*(?:where|select|orderby|groupby|count|union|except|intersect|limit)\s*\("
    if re.search(sr_keywords, response):
        content = re.sub(r"^```\s*", "", response.strip())
        content = re.sub(r"\s*```$", "", content)
        return content.strip()

    return ""


# ============ 准备 Prompt 列表 ============

def prepare_prompts(dataset, table_json, start_idx, end_idx, existing_ids):
    """
    预构建所有 prompt，返回列表。
    每个元素: (qid, prompt_text, item_info_dict)
    """
    prompt_list = []
    skipped = []

    for i in range(start_idx, end_idx):
        qid = i
        if qid in existing_ids:
            skipped.append(qid)
            continue

        item = dataset[i]
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]

        try:
            schema_list = get_column_table_spider(table_json, item, gold_sql)
            if not schema_list:
                # 回退
                schema_list = _get_schema_fallback(table_json, item, gold_sql)
            schema_str = format_schema_for_prompt(schema_list)
        except Exception as e:
            logger.error(f"Schema extraction error for idx={i}, db={db_id}: {e}")
            continue

        prompt = SQL2SR_PROMPT.format(
            shots=SPIDER_SQL2SR_SHOTS,
            question=question,
            schema=schema_str,
            evidence="",
            sql=gold_sql,
        )

        prompt_list.append({
            "qid": qid,
            "item": item,
            "prompt": prompt,
            "schema_list": schema_list,
            "schema_str": schema_str,
        })

    logger.info(
        f"Prepared {len(prompt_list)} prompts "
        f"(skipped {len(skipped)} already done), range [{start_idx}, {end_idx})"
    )
    return prompt_list, skipped


def _get_schema_fallback(table_json, question_info, sql):
    """回退：简单的 schema 提取"""
    db_id = question_info['db_id']
    for content in table_json:
        if content['db_id'] == db_id:
            table_names_list = content["table_names_original"]
            column_names_list = [
                [table_names_list[int(c[0])], c[1]]
                for c in content['column_names_original'][1:]
            ]
            pure_columns = [i[1] for i in column_names_list]
            filtered = [c for c in column_names_list if c[1] in sql]
            return filtered
    return []


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="[Concurrent] Generate SHARE-style SR trajectory")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output", type=str, default=OUTPUT_FILE)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Concurrent workers (default: {MAX_WORKERS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size per round (default: {BATCH_SIZE})")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES,
                        help=f"Max retries per request (default: {MAX_RETRIES})")
    args = parser.parse_args()

    # 加载数据
    logger.info(f"Loading: {TRAIN_FILE}")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Loading: {TABLES_FILE}")
    with open(TABLES_FILE, "r", encoding="utf-8") as f:
        table_json = json.load(f)

    total = len(dataset)
    logger.info(f"Total samples: {total}")

    if args.end < 0:
        args.end = total
    args.end = min(args.end, total)

    output_file = args.output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 断点续传
    existing_results = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        logger.info(f"Resuming from {len(existing_results)} existing results")

    existing_ids = {r["question_id"] for r in existing_results}

    # 预构建所有 prompt
    prompt_list, skipped = prepare_prompts(
        dataset, table_json, args.start, args.end, existing_ids
    )

    if not prompt_list:
        logger.info("No new prompts to process. Done!")
        return

    # 并发批处理
    success_count = len(existing_results)
    fail_count = len(skipped)  # 已完成的 skip 不算失败，重新计数
    fail_count = 0
    total_processed = 0

    best_workers = min(args.workers, estimate_best_workers(MAX_MODEL_LEN))
    logger.info(f"Concurrency: {best_workers} workers, batch_size={args.batch_size}")

    batch_start = time.perf_counter()

    # 分批处理
    for batch_idx in range(0, len(prompt_list), args.batch_size):
        batch = prompt_list[batch_idx:batch_idx + args.batch_size]
        batch_num = batch_idx // args.batch_size + 1
        total_batches = (len(prompt_list) + args.batch_size - 1) // args.batch_size

        # 构建 [(idx, messages)] 格式
        api_inputs = [
            (entry["qid"], [{"role": "user", "content": entry["prompt"]}])
            for entry in batch
        ]

        t0 = time.perf_counter()
        logger.info(
            f"\n[Batch {batch_num}/{total_batches}] "
            f"{len(batch)} prompts, {best_workers} workers..."
        )

        results = batch_chat_with_retry(
            api_inputs,
            max_workers=best_workers,
            max_retries=args.retries,
            timeout=REQUEST_TIMEOUT,
            temperature=0.0,
        )

        batch_elapsed = time.perf_counter() - t0

        # 处理结果
        batch_success = 0
        batch_fail = 0

        # 建立 qid -> entry 映射
        entry_map = {e["qid"]: e for e in batch}

        for qid, content, info in results:
            entry = entry_map[qid]
            item = entry["item"]

            if info["success"] and content:
                sr_trajectory = extract_sr_from_response(content)
                result = {
                    "question_id": qid,
                    "db_id": item["db_id"],
                    "question": item["question"],
                    "gold_sql": item["query"],
                    "trajectory": sr_trajectory,
                    "schema": entry["schema_list"],
                    "schema_str": entry["schema_str"],
                    "raw_response": content,
                }
                existing_results.append(result)
                batch_success += 1
                success_count += 1
            else:
                logger.warning(f"  idx={qid} FAILED after {args.retries} retries")
                batch_fail += 1
                fail_count += 1

        total_processed += len(batch)
        total_tokens = sum(len(r.get("raw_response", "")) for r in existing_results[-batch_success:])

        logger.info(
            f"  Batch done: {batch_elapsed:.1f}s | "
            f"OK={batch_success} FAIL={batch_fail} | "
            f"吞吐≈{len(batch) / batch_elapsed:.1f} req/s"
        )

        # 保存检查点
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)
        logger.info(
            f"  Saved. Progress: {success_count}/{args.end - args.start} "
            f"({100 * success_count / max(1, args.end - args.start):.1f}%)"
        )

    # 最终保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=2)

    total_elapsed = time.perf_counter() - batch_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Success: {success_count} | Failed: {fail_count} | Skipped: {len(skipped)}")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info(f"Avg throughput: {total_processed / total_elapsed:.1f} req/s")
    logger.info(f"Output: {output_file}")


if __name__ == "__main__":
    main()
