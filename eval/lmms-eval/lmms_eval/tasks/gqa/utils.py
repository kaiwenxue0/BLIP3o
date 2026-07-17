import os
from pathlib import Path

import datasets
from datasets import load_dataset
from tqdm import tqdm


_LOCAL_GQA_DATASET_PATH = Path(os.environ.get("GQA_DATASET_PATH", "/home/notebook/code/group/xuekaiwen/data/lmms-lab/GQA"))
_GQA_ID2IMAGE = None


def _load_gqa_split(config_name: str, split: str) -> datasets.Dataset:
    local_split_dir = _LOCAL_GQA_DATASET_PATH / config_name
    if local_split_dir.is_dir():
        return load_dataset(
            str(_LOCAL_GQA_DATASET_PATH),
            data_files={split: f"{config_name}/{split}-*.parquet"},
            split=split,
            verification_mode="no_checks",
        )
    return load_dataset("lmms-lab/GQA", config_name, split=split, token=True)


def gqa_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    gqa_raw_image_dataset = _load_gqa_split("testdev_balanced_images", "testdev")
    gqa_id2image = {}
    for row in tqdm(gqa_raw_image_dataset, desc="Loading GQA images"):
        gqa_id2image[row["id"]] = row["image"].convert("RGB")

    def _process_doc(doc):
        return {
            "image": gqa_id2image[doc["imageId"]],
        }

    return dataset.map(_process_doc, num_proc=8)


def _gqa_id2image():
    global _GQA_ID2IMAGE
    if _GQA_ID2IMAGE is None:
        gqa_raw_image_dataset = _load_gqa_split("testdev_balanced_images", "testdev")
        _GQA_ID2IMAGE = {}
        for row in tqdm(gqa_raw_image_dataset, desc="Loading GQA images"):
            _GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    return _GQA_ID2IMAGE


def gqa_doc_to_visual(doc):
    return [_gqa_id2image()[doc["imageId"]]]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"
