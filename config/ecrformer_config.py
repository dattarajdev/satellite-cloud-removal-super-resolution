from typing import Any

from .base_config import BaseConfig


class ECRformerConfig(BaseConfig):

    def __init__(self, **kwargs: Any):

        super().__init__(**kwargs)

        self.net.name = "ecrformer"

        self.net.cfg = dict(
            features_start=48,
            num_blocks=[2, 3, 2, 2],
            block_type=["ecrformer", "ecrformer"],
            cbam="1ca2+1sa2",
            bottle_neck="tsa",
            num_refine=4,
            pos_encoding=None,
            drop_path_rate=0.0,

            in_chans=[2, 13],
            out_chans=13,
        )

        self.net.output = ["target"]

        self.train.proj_weight = [
            0.05,
            0.05,
        ]

        self.train.train_bs = 4

        self.optim.accumulate_grad_batches = 4