# %%
from hfcnn.models import model_class
from hfcnn import dataset, config
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import logging
import torcheck
import os 

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch_model = model_class.Net()
    torch_model.load_state_dict(torch.load(options['model_path']), strict=False)
    torch_model.to(device)

    # Import the data
    training_data = dataset.HeatLoadDataset(options["train_df_path"])
    logging.info(f"Imported {training_data.__len__()} images from the training data set")
    print(f"Imported {training_data.__len__()} images from the training data set")
    train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True)

    # test_data = dataset.HeatLoadDataset(options["test_df_path"])
    # logging.info(f"Imported {test_data.__len__()} images from the test data set")
    # print(f"Imported {test_data.__len__()} images from the test data set")
    # test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True)

    # if os.path.isfile(options["validation_df_path"]):
    #     val_data = dataset.HeatLoadDataset(options["validation_df_path"])
    #     logging.info(f"Imported {val_data.__len__()} images from the validation data set")
    #     print(f"Imported {val_data.__len__()} images from the validation data set")
    #     val_dataloader = DataLoader(val_data, batch_size=5, shuffle=True)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in torch_model.state_dict():
        print(param_tensor, "\t", torch_model.state_dict()[param_tensor].size())

    # Initialize optimizer
    criterion = nn.L1Loss()
    optimizer = optim.SGD(torch_model.parameters(), lr=0.001, momentum=0.9)

    # torcheck is a tool for preforming some sanity checked during training process.
    # they will return errors if the following checks are violated.
    torcheck.register(optimizer)

    # check to see that model parameters change during training
    torcheck.add_module_changing_check(torch_model, module_name="my_model")

    # check whether model parameters become NaN or outputs contain NaN
    torcheck.add_module_nan_check(torch_model)

    # check whether model parameters become infinite or outputs contain infinite value
    torcheck.add_module_inf_check(torch_model)

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['label']

            # Transfer to GPU
            inputs, labels = inputs.to(device), labels.to(device)

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
            
            del inputs, labels



    print('Finished Training')

    torch.save(torch_model.state_dict(), options['model_path'])

if __name__ == "__main__":
    main()