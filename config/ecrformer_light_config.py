from typing import Any

from .ecrformer_config import ECRformerConfig


class ECRformerLightConfig(ECRformerConfig):

    def __init__(self, **kwargs: Any):

        super().__init__(**kwargs)

        self.net.cfg["features_start"] = 32

        self.net.cfg["num_blocks"] = [
            2,
            2,
            1,
            1,
        ]

        self.net.cfg["num_refine"] = 2

        self.train.train_bs = 8

        self.optim.accumulate_grad_batches = 2