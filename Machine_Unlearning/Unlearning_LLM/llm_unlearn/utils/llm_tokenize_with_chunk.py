import copy
import numpy as np
import torch
from llm_unlearn.utils import compute_logits_and_samples_for_batch
from transformers import BatchEncoding
from datasets import Dataset, DatasetDict
from tqdm import trange

def tokenize(
    dataset,
    tokenizer,
    max_length,
    random_label=False,
    completely_random=False,
    top_k=50,
    top_p=1.0,
    rm_groundtruth=False,
):
    column_names = ["text",]
    text_column_name = "text"

    def chunk_and_pad(examples):
        result = {"input_ids": [], "attention_mask": []}

        for i in range(len(examples["input_ids"])):
            input_ids = examples["input_ids"][i]
            attention_mask = examples["attention_mask"][i]

            num_chunks = len(input_ids) // max_length + int(
                len(input_ids) % max_length != 0
            )
            # num_chunks = len(input_ids) // max_length
            for j in range(num_chunks):
                start_index = j * max_length
                end_index = start_index + max_length

                chunk_input_ids = input_ids[start_index:end_index]
                chunk_attention_mask = attention_mask[start_index:end_index]
                if len(chunk_input_ids) < 5:
                    continue
                padding_length = max_length - len(chunk_input_ids)
                chunk_input_ids= (
    [tokenizer.pad_token_id] * padding_length
    + chunk_input_ids
)
                chunk_attention_mask= (
    [0] * padding_length
    + chunk_attention_mask
)

                result["input_ids"].append(chunk_input_ids)
                result["attention_mask"].append(chunk_attention_mask)

        return result

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        output = chunk_and_pad(output)
        output = BatchEncoding(output, tensor_type="pt")
        output["labels"] = copy.deepcopy(output["input_ids"])
        if random_label:
            if completely_random:
                special_tokens = [
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                    tokenizer.bos_token_id,
                    tokenizer.unk_token_id,
                ]
                for j, sequence in enumerate(output["input_ids"]):
                    for i, token_id in enumerate(sequence):
                        if token_id not in special_tokens:
                            output["labels"][j][i] = np.random.choice(
                                tokenizer.vocab_size
                            )
                        else:
                            output["labels"][j][i] = token_id
            else:
                for i in range(len(output["input_ids"])):
                    input = {
                        key: value[i].unsqueeze(0) for key, value in output.items()
                    }
                    input = BatchEncoding(input, tensor_type="pt")
                    _, sampled_token_ids = compute_logits_and_samples_for_batch(
                        input,
                        tokenizer,
                        top_k=top_k,
                        top_p=top_p,
                        rm_groundtruth=rm_groundtruth,
                    )
                    sampled_token_ids = sampled_token_ids.squeeze(0)
                    indices = torch.nonzero(
                        output["input_ids"][i] == tokenizer.pad_token_id, as_tuple=True
                    )

                    sampled_token_ids[indices] = tokenizer.pad_token_id
                    output["labels"][i] = sampled_token_ids

        pad_token_mask = output["labels"] == tokenizer.pad_token_id
        output["labels"] = torch.where(
            pad_token_mask,
            torch.tensor(-100, device=output["labels"].device),
            output["labels"],
        )
        return output

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        # batch_size=1,
        desc="Running tokenizer",
        load_from_cache_file=False,
        # num_proc=100
    )





def tokenize_token(
    dataset,
    tokenizer,
    max_length,
    target_weight=3.0,
    random_label=False,
    completely_random=False,
    top_k=50,
    top_p=1.0,
    rm_groundtruth=False,
):
    """
    与 tokenize() 基本一致。
    额外返回:
        loss_weight
    """

    column_names = [
        "text",
        "target_spans",
    ]

    text_column_name = "text"

    ########################################################
    # chunk
    ########################################################

    def chunk_and_pad(examples):

        result = {
            "input_ids": [],
            "attention_mask": [],
            "loss_weight": [],
        }

        for i in range(len(examples["input_ids"])):

            input_ids = examples["input_ids"][i]
            attention_mask = examples["attention_mask"][i]
            loss_weight = examples["loss_weight"][i]

            num_chunks = len(input_ids) // max_length + int(
                len(input_ids) % max_length != 0
            )

            for j in range(num_chunks):

                start = j * max_length
                end = start + max_length

                ids = input_ids[start:end]
                mask = attention_mask[start:end]
                weight = loss_weight[start:end]

                if len(ids) < 5:
                    continue

                pad = max_length - len(ids)

                ids = [tokenizer.pad_token_id] * pad + ids
                mask = [0] * pad + mask
                weight = [0.0] * pad + weight

                result["input_ids"].append(ids)
                result["attention_mask"].append(mask)
                result["loss_weight"].append(weight)

        return result

    ########################################################
    # tokenize
    ########################################################

    def tokenize_function(examples):

        output = tokenizer(
            examples[text_column_name],
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        ####################################################
        # build loss weight
        ####################################################

        all_loss_weight = []

        for idx in range(len(output["input_ids"])):

            offsets = output["offset_mapping"][idx]

            weight = [1.0] * len(offsets)

            spans = examples["target_spans"][idx]

            for token_idx, (s, e) in enumerate(offsets):

                if s == e:
                    continue

                for span in spans:

                    span_start = span["start"]
                    span_end = span["end"]

                    if s < span_end and e > span_start:

                        weight[token_idx] = target_weight


            all_loss_weight.append(weight)

        output["loss_weight"] = all_loss_weight

        ####################################################

        output.pop("offset_mapping")

        output = chunk_and_pad(output)

        output = BatchEncoding(output, tensor_type="pt")

        ####################################################
        # labels
        ####################################################

        output["labels"] = copy.deepcopy(output["input_ids"])

        ####################################################
        # RLFT
        ####################################################

        if random_label:

            if completely_random:

                special_tokens = [
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                    tokenizer.bos_token_id,
                    tokenizer.unk_token_id,
                ]

                for j, sequence in enumerate(output["input_ids"]):

                    for i, token_id in enumerate(sequence):

                        if token_id not in special_tokens:

                            output["labels"][j][i] = np.random.choice(
                                tokenizer.vocab_size
                            )

            else:

                for i in range(len(output["input_ids"])):

                    inp = {
                        key: value[i].unsqueeze(0)
                        for key, value in output.items()
                        if key != "loss_weight"
                    }

                    inp = BatchEncoding(inp, tensor_type="pt")

                    _, sampled = compute_logits_and_samples_for_batch(
                        inp,
                        tokenizer,
                        top_k=top_k,
                        top_p=top_p,
                        rm_groundtruth=rm_groundtruth,
                    )

                    sampled = sampled.squeeze(0)

                    indices = torch.nonzero(
                        output["input_ids"][i] == tokenizer.pad_token_id,
                        as_tuple=True,
                    )

                    sampled[indices] = tokenizer.pad_token_id

                    output["labels"][i] = sampled

        ####################################################
        # ignore pad
        ####################################################

        pad_mask = output["labels"] == tokenizer.pad_token_id

        output["labels"] = torch.where(
            pad_mask,
            torch.tensor(-100, device=output["labels"].device),
            output["labels"],
        )

        return output

    ########################################################

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer (token-weight)",
        load_from_cache_file=False,
    )