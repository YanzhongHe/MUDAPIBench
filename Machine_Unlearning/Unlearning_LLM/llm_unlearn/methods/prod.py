import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer
from torch.utils.data import SequentialSampler


def top_p_filtering(
    logits,
    top_p=0.9,
    filter_value=-float("inf"),
    N=1,
    max_N=10,
):
    """
    logits: [B, L, V]
    """

    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(
        probs,
        descending=True,
        dim=-1,
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    sorted_indices_to_remove = cumulative_probs > top_p

    sorted_indices_to_remove[..., 1:] = (
        sorted_indices_to_remove[..., :-1].clone()
    )

    sorted_indices_to_remove[..., 0] = False

    if max_N is None:
        max_N = probs.size(-1)

    max_mask = (
        torch.arange(
            sorted_probs.size(-1),
            device=logits.device,
        )
        >= max_N
    )

    sorted_indices_to_remove |= max_mask

    remove_mask = torch.zeros_like(
        probs,
        dtype=torch.bool,
    )

    remove_mask.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )

    filtered_logits = logits.masked_fill(
        remove_mask,
        filter_value,
    )

    return filtered_logits


def compute_prod_loss(
    pretrained_model,
    current_model,
    batch,
    top_p=0.8,
    alpha=0.0,
    temperature=0.8,
    N=1,
    max_N=10,
):
    device = current_model.device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    ref_logits = pretrained_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    probs = F.softmax(ref_logits, dim=-1)

    labels = labels[..., 1:]
    copied_logits = ref_logits[..., :-1, :].clone()
    vocab_size = copied_logits.size(-1)

    mask_start_pos = 1
    mask_start = torch.zeros_like(labels, dtype=torch.bool)
    mask_start[:, mask_start_pos:] = True

    labels = labels.long()
    ignore_mask = (labels == -100)
    labels[ignore_mask] = 0

    neg_mask = labels < 0
    over_mask = labels >= vocab_size
    if torch.any(neg_mask) or torch.any(over_mask):
        print("=" * 60)
        print(f"vocab_size = {vocab_size}")
        print(f"Number of negative token IDs: {neg_mask.sum().item()}")
        print(f"Number of out-of-vocabulary token IDs: {over_mask.sum().item()}")
        bad_tokens = torch.unique(labels[neg_mask | over_mask])
        print(f"Invalid token ID list: {bad_tokens.tolist()}")
        print("=" * 60)

    labels_safe = torch.clamp(labels, min=0, max=vocab_size - 1)
    valid_mask = mask_start & (~ignore_mask)
    total_valid = valid_mask.sum().item()
    print(f"Total valid answer tokens in the current batch: {total_valid}")

    if total_valid == 0:
        print("Warning: No valid answer tokens in the current batch. Skipping gradient update.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    mask = (
        F.one_hot(labels_safe, num_classes=vocab_size).bool()
        & valid_mask.unsqueeze(-1)
    )
    copied_logits = copied_logits.masked_fill(mask, -float("inf"))

    filtered_logits = top_p_filtering(
        copied_logits,
        top_p=top_p,
        filter_value=-float("inf"),
        N=N,
        max_N=max_N,
    )

    if temperature is None:
        scaled_logits = filtered_logits
    else:
        scaled_logits = filtered_logits / temperature

    ground_truth_distribution = F.softmax(scaled_logits, dim=-1)
    one_hot = F.one_hot(
        labels_safe,
        num_classes=probs.size(-1),
    ).bool()

    ground_truth_distribution = torch.where(
        one_hot,
        -alpha * probs[..., :-1, :],
        ground_truth_distribution,
    )

    model_logits = current_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    model_distribution = F.softmax(model_logits, dim=-1)
    model_distribution = model_distribution[..., :-1, :].contiguous()

    loss = -torch.sum(
        ground_truth_distribution * torch.log(model_distribution + 1e-10),
        dim=-1,
    )
    loss = loss.mean()

    return loss


class PRODTrainer(Trainer):
    """
    HuggingFace Trainer implementation for
    Preference-based Random Output Distillation (PROD).
    """

    def __init__(
        self,
        pretrain_model=None,
        top_p=0.8,
        temperature=0.8,
        N=1,
        max_N=10,
        alpha=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if pretrain_model is None:
            raise ValueError("pretrain_model must not be None.")

        device = self.accelerator.device

        pretrain_model.to(device)
        pretrain_model.eval()

        for p in pretrain_model.parameters():
            p.requires_grad = False

        self.pretrain_model = pretrain_model

        self.top_p = top_p
        self.temperature = temperature
        self.alpha = alpha
        self.max_N = max_N
        self.N = N

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        """
        Compute PROD loss.
        """

        loss = compute_prod_loss(
            pretrained_model=self.pretrain_model,
            current_model=model,
            batch=inputs,
            top_p=self.top_p,
            alpha=self.alpha,
            temperature=self.temperature,
            N=self.N,
            max_N=self.max_N,
        )

        if return_outputs:
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            return loss, outputs

        return loss

    def _get_train_sampler(self):
        """
        Keep the dataset order fixed.

        This follows the same behavior as the GA/GD/NPO implementations.
        """
        return SequentialSampler(self.train_dataset)

    def _set_signature_columns_if_needed(self):
        """
        Keep all columns required by the model.

        This is identical to the implementation used
        in other unlearning trainers.
        """

        if self._signature_columns is None:

            signature = inspect.signature(
                self.model.forward
            )

            self._signature_columns = list(
                signature.parameters.keys()
            )

            self._signature_columns += list(
                set(
                    [
                        "label",
                        "label_ids",
                    ]
                    + self.label_names
                )
            )