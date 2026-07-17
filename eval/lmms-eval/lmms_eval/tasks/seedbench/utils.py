import json
from collections import defaultdict
from math import floor

from loguru import logger as eval_logger


def _stable_bucket_priority(name):
    encoded = str(name).encode("utf-8")
    return sum((index + 1) * byte for index, byte in enumerate(encoded))


def _select_stratified_indices(dataset, *, key_fn, sample_size, min_per_bucket=1):
    total_count = len(dataset)
    if sample_size >= total_count:
        return list(range(total_count))

    buckets = defaultdict(list)
    for index, row in enumerate(dataset):
        buckets[str(key_fn(row))].append(index)

    bucket_names = sorted(buckets)
    if sample_size < len(bucket_names):
        min_per_bucket = 0

    selected_counts = {name: min(min_per_bucket, len(buckets[name])) for name in bucket_names}

    remainders = []
    for name in bucket_names:
        available = len(buckets[name]) - selected_counts[name]
        if available <= 0:
            remainders.append((0.0, _stable_bucket_priority(name), name))
            continue
        proportional = sample_size * len(buckets[name]) / total_count
        raw_target = max(0.0, proportional - selected_counts[name])
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


def seed_process_docs_fast200(dataset):
    selected_indices = _select_stratified_indices(
        dataset,
        key_fn=lambda row: (row.get("data_type", "unknown"), row.get("question_type_id", "unknown")),
        sample_size=200,
        min_per_bucket=1,
    )
    eval_logger.info(
        "SEED-Bench fast subset enabled: selected {} / {} examples stratified by data_type and question_type_id.",
        len(selected_indices),
        len(dataset),
    )
    return dataset.select(selected_indices)


def seed_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["image"]]


def seed_doc_to_text(doc):
    question = doc["question"]
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return f"{question}\nAnswer with the option's letter from the given choices directly."


def seed_process_result(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]
    data_type = doc["data_type"]

    return {f"seed_{data_type}": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}, f"seed_all": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}}


def seed_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"].lower().strip() == result["answer"].lower().strip():
            total_correct += 1
        total_count += 1
    return total_correct / total_count


def seed_aggregation_result_all(results):
    score = seed_aggregation_result(results)
    stored_results = []
    for result in results:
        stored_results.append({"question_id": result["question_id"], "prediction": result["pred"]})
    with open("./seed_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for seed_submission ...")

    return score


def seed_doc_to_text_mc(doc):
    question = doc["question"]
    return f"{question} Answer :"


def seed_doc_to_choice(doc):
    return [doc["choice_a"], doc["choice_b"], doc["choice_c"], doc["choice_d"]]


def seed_doc_to_mc_target(doc):
    answer2choice = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}
    return doc[answer2choice[doc["answer"]]]
