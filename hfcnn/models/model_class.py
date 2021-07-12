import torch.nn as nn
import torch.functional as F
from hfcnn.models import image_claasfication_base


class Net(image_claasfication_base.ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(1, 16, kernel_size = 12, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16,32, kernel_size = 12, stride = 6, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            # nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            # nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            # nn.MaxPool2d(2,2),
            
            # nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            # nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            # nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(89024, 1),
            # nn.ReLU(),
            # nn.Linear(1, 64),
            # nn.ReLU(),
            nn.Linear(1,1)
        )
    
    def forward(self, x):
        for layer in self.network:
            x = layer(x)
            # print(x.size())
        return x

    # def forward(self, xb):
    #     return self.network(xb)