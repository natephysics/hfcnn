import torch.nn as nn
from hfcnn.models import model_class
from torch import Tensor


class HF_Model(model_class.ImageClassificationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 12, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16,32, kernel_size = 12, stride = 6, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.LazyLinear(20),
            # nn.Linear(89024, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
    
    def _forward(self, x: Tensor) -> Tensor:
        y = self.network(x)
        for layer in self.network:
            x = layer(x)
            # print(x.size())
        return y