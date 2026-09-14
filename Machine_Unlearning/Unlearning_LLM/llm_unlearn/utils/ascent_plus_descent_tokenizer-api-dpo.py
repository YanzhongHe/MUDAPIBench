import torch
import json
import os
from transformers import AutoTokenizer, set_seed
from datasets import Dataset

from llm_unlearn.utils import tokenize


# =========================
# 配置
# =========================

MODEL_PATH = "../models/Qwen2.5-Coder-3B"
DATASET_PATH = "../data/qwen-forget-idk.json"
OUTPUT_DIR = "../tokenized_dataset"
MAX_LEN = 4096


# =========================
# 加载数据（拆成 forget / retain）
# =========================
def load_dataset_custom(path):

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    forget_data = []
    retain_data = []

    for x in raw:
        case_id = str(x.get("case-id", ""))

        bad_func = str(x.get("bad function", ""))
        good_func = str(x.get("function", ""))

        # forget：bad function
        forget_data.append({
            "case-id": case_id,
            "text": bad_func
        })

        # retain：good function
        retain_data.append({
            "case-id": case_id,
            "text": good_func
        })

    return Dataset.from_list(forget_data), Dataset.from_list(retain_data)


# =========================
# 过滤长样本
# =========================
def filter_long_samples(dataset, tokenizer):

    def add_length(example):
        example["length"] = len(tokenizer(example["text"])["input_ids"])
        return example

    dataset = dataset.map(add_length)
    dataset = dataset.filter(lambda x: x["length"] <= MAX_LEN)
    dataset = dataset.remove_columns(["length"])

    return dataset


# =========================
# Adv Dataset（直接复用你原逻辑）
# =========================
class AdvSupervisedDataset(torch.utils.data.Dataset):

    def __init__(self, negative_data, positive_data, positive_ratio=1, positive_factor=1.0):

        negative_data = negative_data.to_dict()
        positive_data = positive_data.to_dict()

        self.input_ids = []
        self.labels = []
        self.attention_mask = []
        self.factor = []

        for i in range(len(negative_data["input_ids"])):

            # negative（forget）
            self.input_ids.append(negative_data["input_ids"][i])
            self.labels.append(negative_data["labels"][i])
            self.attention_mask.append(negative_data["attention_mask"][i])
            self.factor.append(-1)

            # positive（retain）
            self.input_ids.extend(
                positive_data["input_ids"][
                    i * positive_ratio : (i + 1) * positive_ratio
                ]
            )
            self.labels.extend(
                positive_data["labels"][
                    i * positive_ratio : (i + 1) * positive_ratio
                ]
            )
            self.attention_mask.extend(
                positive_data["attention_mask"][
                    i * positive_ratio : (i + 1) * positive_ratio
                ]
            )
            self.factor.extend([positive_factor] * positive_ratio)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i],
            "factor": self.factor[i],
        }


# =========================
# 主流程
# =========================
def run_adv_dataset(positive_ratio=1, positive_factor=1.0):

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        padding_side="left",
        trust_remote_code=True,
        model_max_length=MAX_LEN,
    )

    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.pad_token = tokenizer.eos_token

    # =========================
    # load ds-new
    # =========================
    forget_dataset, retain_dataset = load_dataset_custom(DATASET_PATH)

    # =========================
    # filter
    # =========================
    forget_dataset = filter_long_samples(forget_dataset, tokenizer)
    retain_dataset = filter_long_samples(retain_dataset, tokenizer)

    # =========================
    # tokenize
    # =========================
    forget_dataset = tokenize(forget_dataset, tokenizer, MAX_LEN)
    retain_dataset = tokenize(retain_dataset, tokenizer, MAX_LEN)

    # =========================
    # 构造 adv dataset
    # =========================
    train_dataset = AdvSupervisedDataset(
        forget_dataset,
        retain_dataset,
        positive_ratio=positive_ratio,
        positive_factor=positive_factor,
    )

    # =========================
    # save
    # =========================
    save_path = os.path.join(
        OUTPUT_DIR,
        "qwen-forget",
        "ascent_plus_descent"
    )

    os.makedirs(save_path, exist_ok=True)

    save_file = os.path.join(save_path, "tokenized_dataset.pt")
    torch.save(train_dataset, save_file)

    print(f"✅ Saved to: {save_file}")
    print(f"📊 Final samples: {len(train_dataset)}")


# =========================
# main
# =========================
if __name__ == "__main__":

    run_adv_dataset(
        positive_ratio=1,
        positive_factor=1.0
    )