import inspect
from typing import Optional

import torch
from transformers import Trainer
from torch.utils.data import SequentialSampler

from .utils import compute_dpo_loss

import torch


class DPODataCollator:


    def __call__(self, features):


        batch = {}


        keys = [

            "win_input_ids",
            "win_attention_mask",
            "win_labels",

            "lose_input_ids",
            "lose_attention_mask",
            "lose_labels",

        ]


        for key in keys:

            batch[key] = torch.tensor(
                [
                    f[key]
                    for f in features
                ],
                dtype=torch.long
            )


        batch["case-id"] = [
            f["case-id"]
            for f in features
        ]


        return batch

class DPOTrainer(Trainer):
    """
    """

    def __init__(
        self,
        pretrain_model=None,
        beta=1,
        gamma=1.0,
        alpha=1.0,
        **kwargs,
    ):

        super().__init__(**kwargs)


        if pretrain_model is None:
            raise ValueError(
                "pretrain_model must not be None."
            )


        device = self.accelerator.device


        # reference model
        pretrain_model.to(device)
        pretrain_model.eval()


        for p in pretrain_model.parameters():
            p.requires_grad = False
        
        
        pretrain_model.config.use_cache = False
        pretrain_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )


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
        # Construct win / lose inputs
        ########################################


        win_inputs = {

            "input_ids":
                inputs["win_input_ids"],

            "attention_mask":
                inputs["win_attention_mask"],

            "labels":
                inputs["win_labels"],
        }



        lose_inputs = {

            "input_ids":
                inputs["lose_input_ids"],

            "attention_mask":
                inputs["lose_attention_mask"],

            "labels":
                inputs["lose_labels"],
        }



        ########################################
        # DPO Loss
        ########################################


        dpo_loss, outputs = compute_dpo_loss(

            model=model,

            ref_model=self.pretrain_model,

            win_inputs=win_inputs,

            lose_inputs=lose_inputs,

            beta=self.beta,
        )



        loss = (
            dpo_loss

        )



        return (
            (loss, outputs)
            if return_outputs
            else loss
        )



    def _get_train_sampler(
        self
    ) -> Optional[torch.utils.data.Sampler]:

        return SequentialSampler(
            self.train_dataset
        )



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
                    [
                        "label",
                        "label_ids"
                    ]
                    +
                    self.label_names
                )

            )


            ################################
            # keep DPO columns
            ################################

            self._signature_columns += [

                "case-id",

                "win_input_ids",
                "win_attention_mask",
                "win_labels",

                "lose_input_ids",
                "lose_attention_mask",
                "lose_labels"
            ]