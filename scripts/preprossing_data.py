# %%
import argparse
from hfcnn.lib import network_configuration, dataset
import logging

logging.basicConfig(
    filename='./logs/preprossing_data.txt', 
    filemode='a', 
    format='%(asctime)s %(msecs)d- %(process)d -%(levelname)s - %(message)s', 
    datefmt='%d-%b-%y %H:%M:%S %p' ,
    level=logging.DEBUG
    )

def main():
    # grab the path from the argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-training_config_path", help="path to the training config.")
    training_config_path = parser.parse_args().training_config_path

    # generate the configuration
    config = network_configuration.GenerateConfig(training_config_path)
    
    # Step 1. Import the raw the data
    logging.info('Constructing raw data set.')
    raw_data = dataset.HeatLoadDataset(
        config.value('raw_df_path'),
        config.value('raw_data_path'))

    # Step 2. Filter the data
    


    
    print(config.config)

# %%
if __name__ == "__main__":
    main()