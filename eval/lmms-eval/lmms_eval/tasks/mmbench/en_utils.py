import json
import os
import re
from collections import defaultdict
from math import floor
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator

with open(Path(__file__).parent / "mmbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION") or config["metadata"].get("gpt_eval_model_name", "gpt-4o-2024-11-20")
API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
else:
    API_URL = "YOUR_API_URL"
    API_KEY = "YOUR_API_KEY"


mmbench_evaluator = MMBench_Evaluator(sys_prompt=config["metadata"]["sys_prompt"], API_KEY=API_KEY, API_URL=API_URL, model_version=GPT_EVAL_MODEL_NAME)


def _normalize_option_letter(text):
    if text is None:
        return ""
    cleaned = str(text).strip().upper()
    match = re.search(r"\b([A-E])\b", cleaned)
    if match:
        return match.group(1)
    for char in cleaned:
        if char in {"A", "B", "C", "D", "E"}:
            return char
    return ""


def _judge_api_available():
    return API_KEY not in {"", "YOUR_API_KEY"} and API_URL not in {"", "YOUR_API_URL"}


def _local_letter_accuracy(results):
    if not results:
        return 0.0
    correct = 0
    for result in results:
        pred = _normalize_option_letter(result.get("prediction", ""))
        answer = _normalize_option_letter(result.get("answer", ""))
        if pred and answer and pred == answer:
            correct += 1
    return correct / len(results)


def _stable_bucket_priority(name):
    encoded = str(name).encode("utf-8")
    return sum((index + 1) * byte for index, byte in enumerate(encoded))


def _select_stratified_indices(dataset, *, key_name, sample_size, min_per_bucket=1):
    total_count = len(dataset)
    if sample_size >= total_count:
        return list(range(total_count))

    buckets = defaultdict(list)
    for index, row in enumerate(dataset):
        buckets[str(row.get(key_name, "unknown"))].append(index)

    bucket_names = sorted(buckets)
    if sample_size < len(bucket_names):
        min_per_bucket = 0

    selected_counts = {name: min(min_per_bucket, len(buckets[name])) for name in bucket_names}
    remaining = sample_size - sum(selected_counts.values())
    if remaining < 0:
        remaining = 0

    raw_targets = {}
    remainders = []
    for name in bucket_names:
        available = len(buckets[name]) - selected_counts[name]
        if available <= 0:
            raw_targets[name] = 0.0
            remainders.append((0.0, _stable_bucket_priority(name), name))
            continue
        proportional = sample_size * len(buckets[name]) / total_count
        raw_target = max(0.0, proportional - selected_counts[name])
        raw_targets[name] = raw_target
        take_now = min(available, floor(raw_target))
        selected_counts[name] += take_now
        remainder = raw_target - floor(raw_target)
        remainders.append((remainder, _stable_bucket_priority(name), name))

    remaining = sample_size - sum(selected_counts.values())
    for _, _, name in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if selected_counts[name] >= len(buckets[name]):
            continue
        selected_counts[name] += 1
        remaining -= 1

    if remaining > 0:
        for name in bucket_names:
            while remaining > 0 and selected_counts[name] < len(buckets[name]):
                selected_counts[name] += 1
                remaining -= 1

    selected_indices = []
    for name in bucket_names:
        selected_indices.extend(buckets[name][: selected_counts[name]])
    return sorted(selected_indices)


def mmbench_process_docs_fast200(dataset):
    selected_indices = _select_stratified_indices(
        dataset,
        key_name="category",
        sample_size=200,
        min_per_bucket=1,
    )
    eval_logger.info(
        "MMBench fast subset enabled: selected {} / {} examples stratified by category.",
        len(selected_indices),
        len(dataset),
    )
    return dataset.select(selected_indices)


def mmbench_doc_to_visual(doc):
    num_image = int(os.environ.get("NUM_IMAGE", "1"))
    if num_image == 1:
        return [doc["image"].convert("RGB")]
    if num_image == 2:
        image = doc["image"].convert("RGB")
        return [image, image.copy()]
    raise ValueError(f"num_image must be 1 or 2, got {num_image}")


def mmbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, options_dict = mmbench_evaluator.create_options_prompt(doc, option_candidate)

    data = {
        # "img": doc["image"],
        "question": doc["question"],
        "answer": doc.get("answer", None),
        "options": options_prompt,
        "category": doc["category"],
        "L2-category": doc["L2-category"],
        "options_dict": options_dict,
        "index": doc["index"],
        "hint": doc["hint"],
        "source": doc["source"],
        "split": doc["split"],
    }

    query_prompt = f"{data['hint']} {data['question']} {data['options']}" if pd.notna(data["hint"]) and data["hint"] != "nan" else f"{data['question']} {data['options']}"

    if lmms_eval_specific_kwargs:
        query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['post_prompt']}"

    return query_prompt


def mmbench_process_results(doc, results):
    model_response = results[0].strip()
    data = {
        "gpt_eval_score": {
            "index": doc["index"],
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "hint": doc["hint"],
            "source": doc["source"],
            "split": doc["split"],
            "category": doc["category"],
            "L2-category": doc["L2-category"],
        },
        "submission": {
            "index": doc["index"],
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "hint": doc["hint"],
            "source": doc["source"],
            "split": doc["split"],
            "category": doc["category"],
            "L2-category": doc["L2-category"],
        },
    }
    option_candidate = ["A", "B", "C", "D", "E"]
    for c in option_candidate:
        data["submission"][c] = doc.get(c, "nan")
        data["gpt_eval_score"][c] = doc.get(c, "nan")
    return data


def mmbench_aggregate_dev_results_eval(results, args):
    print(f"============= MMBench-EN(Dev) Detailed Results =============")
    if not _judge_api_available():
        overall_acc = _local_letter_accuracy(results)
        eval_logger.warning(
            "MMBench judge credentials are unavailable. Falling back to local option-letter accuracy; "
            "this is useful for AR/MDM comparison but is not the official judge-based score."
        )
        file = generate_submission_file("mmbench_en_dev_results.local_accuracy.json", args)
        with open(file, "w") as f:
            json.dump(
                {
                    "overall_acc": overall_acc,
                    "metric_name": "local_option_letter_accuracy",
                    "official_judge_used": False,
                },
                f,
            )
        return overall_acc * 100
    overall_acc, category_acc, l2_category_acc = mmbench_evaluator.eval_result(results, eval_method="openai")
    file = generate_submission_file("mmbench_en_dev_results.json", args)
    details_info = {
        "overall_acc": overall_acc,
        "category_acc": category_acc,
        "l2_category_acc": l2_category_acc,
    }
    with open(file, "w") as f:
        json.dump(details_info, f)
    return overall_acc * 100


def mmbench_aggregate_dev_results_submission(results, args):
    df = pd.DataFrame(results)
    excel_write_path = generate_submission_file("mmbench_en_dev_results.xlsx", args)
    try:
        with pd.ExcelWriter(excel_write_path) as writer:
            df.to_excel(writer, index=False)
        eval_logger.info(f"Saved results to {excel_write_path}")
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        csv_write_path = generate_submission_file("mmbench_en_dev_results.csv", args)
        df.to_csv(csv_write_path, index=False)
        eval_logger.warning(
            "openpyxl is unavailable; saved MMBench dev submission as CSV instead of XLSX: {}",
            csv_write_path,
        )


def mmbench_aggregate_test_results(results, args):
    df = pd.DataFrame(results)
    excel_write_path = generate_submission_file("mmbench_en_test_results.xlsx", args)
    try:
        with pd.ExcelWriter(excel_write_path) as writer:
            df.to_excel(writer, index=False)
        eval_logger.info(f"Saved results to {excel_write_path}")
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        csv_write_path = generate_submission_file("mmbench_en_test_results.csv", args)
        df.to_csv(csv_write_path, index=False)
        eval_logger.warning(
            "openpyxl is unavailable; saved MMBench test submission as CSV instead of XLSX: {}",
            csv_write_path,
        )
