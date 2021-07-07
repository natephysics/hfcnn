# %%
## TODO-2 setup the config.cfg to house values like data location and model location
from hfcnn.models import model_class
from hfcnn import dataset, config
import torch.optim as optim
import torch.nn as nn
import torch
import logging

# import the options
options = config.construct_options_dict()

logging.basicConfig(
    filename=options['train_log_path'],
    filemode="a",
    force=True,
    format="%(asctime)s %(msecs)d- %(process)d -%(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S %p",
    level=logging.DEBUG,
)

def main():
    device = torch.device("cuda")

    # path to the model
    PATH = './models/init_model.pt'

    torch_model = model_class.Net()
    torch_model.load_state_dict(torch.load(PATH), strict=False)
    torch_model.to(device)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in torch_model.state_dict():
        print(param_tensor, "\t", torch_model.state_dict()[param_tensor].size())

    # Initialize optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(torch_model.parameters(), lr=0.001, momentum=0.9)

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = torch_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

# %%
