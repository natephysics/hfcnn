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
            nn.Linear(89024, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
    
    def _forward(self, x: Tensor) -> Tensor:
        y = self.network(x)
        return y

    # def forward(self, x):
    #     for layer in self.network:
    #         x = layer(x)
    #         # print(x.size())
    #     return x

    # def training_step(self, batch):
    #     # get the inputs; data is a list of [inputs, labels]
    #     inputs, labels = batch['image'], batch['label']
    #     output = self.forward(inputs)
    #     loss = self.loss(output, labels)
    #     return loss

    # def validation_step(self, valid_batch):
    #     inputs, labels = valid_batch['image'], valid_batch['label']
    #     output = self.forward(inputs)
    #     loss = self.loss(output, labels)



    # def configure_optimizers(self):
    #     optimizer = optim.SGD(
    #         self.parameters(), 
    #         lr=model_params['learning_rate'], 
    #         momentum=model_params['momentum']
    #         )
    #     return optimizer

