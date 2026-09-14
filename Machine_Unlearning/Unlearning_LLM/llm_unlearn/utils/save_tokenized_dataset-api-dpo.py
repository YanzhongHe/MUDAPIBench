import torch
import json
import os
from transformers import AutoTokenizer, set_seed
from datasets import Dataset

from llm_unlearn.utils import tokenize


# =========================
# 配置
# =========================
MODEL_PATH = "/openbayes/home/Unlearning_LLM-main/llm_unlearn/models/starcoder2-3b"
DATASET_PATH = "../data/split_data/qwen-forget-idk.json"
OUTPUT_DIR = "../tokenized_dataset"

MAX_LEN = 4096


# =========================
# 加载 + 预处理
# =========================
def load_dataset_custom(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for x in raw:
        data.append({
            "case-id": str(x.get("case-id", "")),
            "text": str(x.get("win_inputs", ""))
        })

    return Dataset.from_list(data)


# =========================
# 过滤超长样本（高性能版）
# =========================
def filter_long_samples(dataset, tokenizer):

    # Step 1: 计算长度（只做一次 tokenize）
    def add_length(example):
        example["length"] = len(tokenizer(example["text"])["input_ids"])
        return example

    dataset = dataset.map(add_length)

    # Step 2: filter
    dataset = dataset.filter(lambda x: x["length"] <= MAX_LEN)

    # Step 3: remove helper column
    dataset = dataset.remove_columns(["length"])

    return dataset


# =========================
# tokenization pipeline
# =========================
def run_tokenization(tokenize_method, completely_random=False, top_k=1, rm_groundtruth=False):

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        padding_side="left",
        trust_remote_code=True,
        model_max_length=MAX_LEN,
    )
    # print("=" * 80)
    # print("tokenizer.padding_side =", tokenizer.padding_side)
    # print("tokenizer.pad_token =", tokenizer.pad_token)
    # print("tokenizer.pad_token_id =", tokenizer.pad_token_id)
    # print("tokenizer.eos_token =", tokenizer.eos_token)
    # print("tokenizer.eos_token_id =", tokenizer.eos_token_id)
    # print("=" * 80)

    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.pad_token = tokenizer.eos_token

    # =========================
    # load data
    # =========================
    dataset = load_dataset_custom(DATASET_PATH)

    # =========================
    # filter long samples
    # =========================
    dataset = filter_long_samples(dataset, tokenizer)

    # =========================
    # save path
    # =========================
    dataset_name = os.path.basename(DATASET_PATH).split(".")[0]
    save_path = os.path.join(OUTPUT_DIR, dataset_name, tokenize_method)

    # =========================
    # tokenize
    # =========================
    if tokenize_method == "normal-DPO":

        dataset = tokenize(dataset, tokenizer, MAX_LEN)
        
    elif tokenize_method == "normal-train":

        dataset = tokenize(dataset, tokenizer, MAX_LEN)        

    elif tokenize_method == "random_label":

        if completely_random:
            dataset = tokenize(
                dataset,
                tokenizer,
                MAX_LEN,
                random_label=True,
                completely_random=True,
            )
            save_path = os.path.join(save_path, "completely_random")

        else:
            dataset = tokenize(
                dataset,
                tokenizer,
                MAX_LEN,
                random_label=True,
                top_k=top_k,
                rm_groundtruth=rm_groundtruth,
            )
            save_path = os.path.join(save_path, f"top_k{top_k}")

    else:
        raise ValueError("Invalid tokenize_method")

    if rm_groundtruth:
        save_path += "_rmgt"

    # =========================
    # save
    # =========================
    os.makedirs(save_path, exist_ok=True)

    save_file = os.path.join(save_path, "tokenized_dataset.pt")
    torch.save(dataset, save_file)

    print(f"✅ Saved to: {save_file}")
    print(f"📊 Final samples: {len(dataset)}")


# =========================
# main
# =========================
if __name__ == "__main__":

    run_tokenization("normal-DPO")