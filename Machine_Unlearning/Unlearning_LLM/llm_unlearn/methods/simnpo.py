import inspect
from typing import Optional

import torch
from transformers import Trainer
from torch.utils.data import SequentialSampler

from .utils import (
    compute_simnpo_loss,
    compute_kl_divergence,
)


class SimNPOTrainer(Trainer):
    """
    """

    def __init__(
        self,
        beta=1.0,
        delta=0.0,
        gamma=1.0,
        alpha=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        device = self.accelerator.device

        self.beta = beta
        self.delta = delta
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
        # Forget Loss (SimNPO)
        ########################################

        if len(inputs["input_ids"]) > 0:

            forget_loss, outputs = compute_simnpo_loss(
                model=model,
                lose_inputs=inputs,
                beta=self.beta,
                delta=self.delta,
            )

        else:

            forget_loss = torch.tensor(
                0.0,
                device=model.device,
            )

        loss = (forget_loss)

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(
        self,
    ) -> Optional[torch.utils.data.Sampler]:

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