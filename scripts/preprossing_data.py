# %%
import argparse
from hfcnn.lib import network_configuration, dataset, filters
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
        config.get('raw_df_path'),
        config.get('raw_data_path')
        )

    # Step 2. Filter the data
    if config.num_of_filters() == 0:
        logging.info('No filters to apply. Skipping step')
    else:
        for filter in config.get('filters_to_apply'):
           logging.info(f'Applying {filter[0]} filters')

           raw_data = raw_data.apply(filters.return_filter(filter[0], filter[1]))

    
    print(config.config)

# %%
if __name__ == "__main__":
    main()