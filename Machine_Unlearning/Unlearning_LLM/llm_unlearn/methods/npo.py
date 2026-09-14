import inspect
from typing import Optional

import torch
from transformers import Trainer
from torch.utils.data import SequentialSampler

from .utils import (
    compute_npo_loss,
    compute_kl_divergence,
)


class NPOTrainer(Trainer):
    """
    """

    def __init__(
        self,
        pretrain_model=None,
        beta=1.0,
        gamma=1.0,
        alpha=1.0,
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

        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):

        outputs = None

        ########################################
        # Forget Loss (NPO)
        ########################################

        if len(inputs["input_ids"]) > 0:

            forget_loss, outputs = compute_npo_loss(
                model=model,
                ref_model=self.pretrain_model,
                lose_inputs=inputs,
                beta=self.beta,
            )

        else:
            print("+++++")

            forget_loss = torch.tensor(
                0.0,
                device=model.device,
            )



        loss = (
            forget_loss
        )

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)

    def _set_signature_columns_if_needed(self):

        if self._signature_columns is None:

            signature = inspect.signature(
                self.model.forward
            )

            self._signature_columns = list(
                signature.parameters.keys()
            )

            self._signature_columns += list(
                set(
                    ["label", "label_ids"]
                    + self.label_names
                )
            )
