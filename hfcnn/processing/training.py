# %%
from hfcnn.models import model_class
from hfcnn import dataset, config, files, yaml_tools
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import torcheck
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from hyperopt import STATUS_OK, STATUS_FAIL
from tqdm import tqdm
import rna
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# import the default options
options = config.construct_options_dict()

# import the model parameters 
model_params = yaml_tools.import_configuration(options['training_config_path'])

# establish logging settings
logging.basicConfig(
    filename=options['log_path'],
    filemode="a",
    force=True,
    format="%(asctime)s %(msecs)d- %(process)d -%(levelname)s -" + str(model_params['session_name']) + " -%(message)s",
    datefmt="%d-%b-%y %H:%M:%S %p",
    level=logging.DEBUG,
)

def main():
    # timestamp start of training loop
    status = STATUS_OK

    # set up tensorboard writer
    folder = os.path.join(options['tensorboard_dir'], model_params['session_name'])
    writer = SummaryWriter(folder)

    options['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set up model
    rna.path.cp(options['untrained_model_path'], options['training_model_path'])


    torch_model = model_class.Net()
    torch_model.load_state_dict(torch.load(options['training_model_path']), strict=False)
    torch_model.to(options['device'])

    # Import the data
    training_data = dataset.HeatLoadDataset(options["train_df_path"])
    logging.info(f"Imported {training_data.__len__()} images from the training data set")
    print(f"Imported {training_data.__len__()} images from the training data set")
    train_dataloader = DataLoader(
        training_data, 
        batch_size=model_params['batch_size'], 
        shuffle=True, 
        pin_memory=torch.cuda.is_available()
        )

    # import the validation data
    val_data = dataset.HeatLoadDataset(options["validation_df_path"])
    logging.info(f"Imported {val_data.__len__()} images from the validation data set")
    print(f"Imported {val_data.__len__()} images from the validation data set")

    if val_data.__len__() < model_params['batch_size']:
        val_batch_size = 10
    else:
        val_batch_size = model_params['batch_size']

    val_dataloader = DataLoader(
        val_data, 
        batch_size=val_batch_size,
        shuffle=True, 
        pin_memory=torch.cuda.is_available()
        )

    # import the test data
    if os.path.isfile(options["test_df_path"]):
        test_data = dataset.HeatLoadDataset(options["test_df_path"])
        logging.info(f"Imported {test_data.__len__()} images from the test data set")
        print(f"Imported {test_data.__len__()} images from the test data set")

        test_dataloader = DataLoader(
            test_data, 
            batch_size=val_batch_size,
            shuffle=True, 
            pin_memory=torch.cuda.is_available()
            )    

    # grab some images for tensorboard
    temp_data = next(iter(train_dataloader))
    temp_data = temp_data['image'].to(options['device'])
    writer.add_graph(torch_model, temp_data)
    del temp_data
    writer.close()
    
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in torch_model.state_dict():
        print(param_tensor, "\t", torch_model.state_dict()[param_tensor].size())

    # Initialize optimizer
    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(
        torch_model.parameters(), 
        lr=model_params['learning_rate'], 
        momentum=model_params['momentum']
        )
    # optimizer = optim.Adam(
    #     torch_model.parameters(), 
    #     lr=model_params['learning_rate']
    #     )

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
    
    best_val_batch_loss = np.inf

    error_count = 0

    for epoch in range(model_params['epochs']):
        torch_model.train()
        loop = tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader))
        for i, data in loop:
            # keep track of i over epochs
            i = i + train_dataloader.__len__() * epoch

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['label']

            # Transfer to GPU
            inputs, labels = inputs.to(options['device']), labels.to(options['device'])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = torch_model(inputs)
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            try:
                optimizer.step()
            except RuntimeError:
                error_count += 1

            del inputs, labels

            # print statistics
            loop.set_description(f"Epoch [{epoch + 1}/{model_params['epochs']}]")
            loop.set_postfix(loss = loss.item())

            writer.add_scalar("Loss", loss.item(), i) 
            
            if i % 50 == 49:
                validation_loss = validate(val_dataloader, torch_model, criterion, 10, **options)
                # validate the network
                writer.add_scalar(
                    "Validation.Loss",
                    validation_loss,
                    i
                    )
                if validation_loss < best_val_batch_loss:
                    best_val_batch_loss = validation_loss
                    torch.save(torch_model.state_dict(), options['best_model_path'])
            
            # break out of both loops if there's too many errors
            max_errors = 10
            if error_count >= max_errors:
                break
        if error_count >= max_errors:
                break

    if error_count > 0:
        logging.error(f'There have beee {error_count} RuntimeErrors recorded. The max errors before break is {max_errors}.')    

    if status == STATUS_OK:
        best_model = model_class.Net()
        best_model.load_state_dict(torch.load(options['best_model_path']), strict=False)
        best_model.to(options['device'])

        # Append the results to a dataframe
        validation_loss = validate(val_dataloader, best_model, criterion, 10, **options)

    else:
        validation_loss = np.nan

    results = {
        'session_name': [model_params['session_name']],
        'validation_loss': [validation_loss],
        'status': [status]
    }

    if os.path.isfile(options["training_results"]):
        training_results = files.import_file_from_local_cache(options["training_results"])
        training_results.append(results, ignore_index=True)
    else:
        training_results = pd.DataFrame(results)
    files.export_data_to_local_cache(training_results, options["training_results"])

    print('Complete')

def validate(val_loader, model, criterion, loop_lim=False, **options):
    valid_loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            val_inputs, val_labels = data['image'], data['label']

            # Transfer to GPU
            val_inputs, val_labels = val_inputs.to(options['device']), val_labels.to(options['device'])

            # forward and get loss
            outputs = model(val_inputs)
            loss = criterion(outputs, val_labels)
        
            # print statistics
            valid_loss += loss.item()
            del val_inputs, val_labels

            if isinstance(loop_lim, int):
                if i == loop_lim:
                    break
    # averaging factor
    if loop_lim == True:
        norm = loop_lim
    else:
        norm = len(val_loader)
        
    return valid_loss / norm


def expected_vs_estimated(test_loader, model, **options):
    model.eval()
    expected = []
    estimated = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            test_inputs, test_labels = data['image'], data['label']

            # Transfer to GPU
            test_inputs, test_labels = test_inputs.to(options['device']), test_labels.to(options['device'])

            # get the predicted values
            outputs = model(test_inputs)

            test_labels = test_labels.cpu().data.numpy().flatten()
            outputs = outputs.cpu().data.numpy().flatten()

            expected = np.append(expected, test_labels)
            estimated = np.append(estimated, outputs)

    return expected, estimated

if __name__ == "__main__":
    main()
# %%
    # expected, estimated = expected_vs_estimated(test_dataloader, best_model, **options)

    # plt.figure(figsize=(10,10))
    # plt.scatter(expected, estimated, c='crimson')
    # p1 = max(max(estimated), max(expected))
    # p2 = min(min(estimated), min(expected))
    # plt.plot([p1, p2], [p1, p2], 'b-')
    # plt.xlabel('True Values', fontsize=15)
    # plt.ylabel('Predictions', fontsize=15)
    # plt.axis('equal')
    # plt.savefig('test.png')
    # plt.show()