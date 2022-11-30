import torch.nn as nn
from hfcnn.models import model_class
from hfcnn.models import blocks
from torch import Tensor
from hydra.utils import instantiate


class HF_Model(model_class.ImageClassificationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=12, padding=1),
            self.act_fn,
            nn.Conv2d(16, 32, kernel_size=12, stride=6, padding=1),
            self.act_fn,
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(20),
            self.act_fn,
            nn.Linear(20, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.network(x)
        for layer in self.network:
            x = layer(x)
            # print(x.size())
        return y


class GoogleNet(model_class.ImageClassificationBase):
    def __init__(self, act_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_fn = instantiate(act_fn)
        self.hparams["network_parameters"] = {
            "act_fn_name": act_fn._target_,
            "act_fn": self.act_fn,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
        self._create_network()
        self._init_params()

    def _create_network(self):
        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.act_fn,
        )
        # Stacking inception blocks
        self.inception_blocks = nn.Sequential(
            blocks.InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.act_fn,
            ),
            blocks.InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 32x32 => 16x16
            blocks.InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.act_fn,
            ),
            blocks.InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.act_fn,
            ),
            blocks.InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.act_fn,
            ),
            blocks.InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
                act_fn=self.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8
            blocks.InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.act_fn,
            ),
            blocks.InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.act_fn,
            ),
        )
        # Output Network
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, self.output_dim)
        )

    def _init_params(self):
        # Initialize the convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x
