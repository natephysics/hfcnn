# build the initial network structure
import torch
from hfcnn.models import model_class

## TODO move the path to the config.cfg'
PATH = './models/init_model.pt'

# instantiate model class
torch_model = model_class.Net()


# Print model's state_dict
print("Model's state_dict:")
for param_tensor in torch_model.state_dict():
    print(param_tensor, "\t", torch_model.state_dict()[param_tensor].size())

torch.save(torch_model.state_dict(), PATH)
