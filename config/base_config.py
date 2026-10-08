from argparse import Namespace
from typing import Any


class BaseConfig(Namespace):

    NUM_CHANS = {
        "SAR": 2,
        "cloudy": 13,
        "target": 13,
    }

    def __init__(self, **kwargs: Any) -> None:

        super().__init__(**kwargs)

        self.seed = 42

        # Dataset
        self.dataset = Namespace(
            name="sen12mscr",
            root="",
            split=["train", "val", "test"],
            train_ratio=0.8,
            data_range=1.0,
            crop_size=128,
        )

        # Training
        self.train = Namespace(
            max_epoch=200,
            early_stop=10,
            lr=4e-4,
            loss_weight=[0.9, 0.1],
            proj_weight=[0.0, 0.0],
            train_bs=8,
            valid_bs=16,
            num_workers=8,
            ckpt_path=None,
            save_dir="./experiments",
        )

        # Optimizer
        self.optim = Namespace(
            accelerator="auto",
            precision=32,
        )

        # Network
        self.net = Namespace(
            name="ecrformer",
            input=["SAR", "cloudy"],
            output=["target"],
            cfg={},
        )

        self.net.cfg["in_chans"] = [
            self.NUM_CHANS[key]
            for key in self.net.input
        ]

        self.net.cfg["out_chans"] = sum(
            self.NUM_CHANS[key]
            for key in self.net.output
        )