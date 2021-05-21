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
    

    #### Step 1. Import the raw the data ####
    raw_data = dataset.HeatLoadDataset(
        config.get('raw_df_path'),
        config.get('raw_data_path')
        )
    logging.info(f'Imported {raw_data.__len__()} images from the raw data set')  
    print(f'Imported {raw_data.__len__()} images from the raw data set')

    #### Step 2. Allpying filter(s) to the data ####
    if config.num_of_filters() == 0:
        logging.info('No filters to apply. Skipping step')
    else:
        for filter in config.get('filters_to_apply'):
           logging.info(f'Applying {filter[0]} filters')
           raw_data = raw_data.apply(filters.return_filter(*filter)) 
           logging.info(f'{raw_data.__len__()} images remain after applying {filter[0]} filter') 
           print(f'{raw_data.__len__()} images remain after applying {filter[0]} filter')   


    #### Step 3. Test/Train/Val Split ####
    # Because data for a given program number is likely corolated we'll divide
    # the sets up by program number.
    
    # generate lists of program numbers for each train/test/val split
    program_num_split = filters.split(raw_data.program_nums(), config.get('train_test_split'))
    
    # generate the training data
    training_data = raw_data.split_by_program_num(program_num_split[0])
    logging.info(f'Training dataset generated with {training_data.__len__()} samples.')

    # generate the test data
    test_data = raw_data.split_by_program_num(program_num_split[1])
    logging.info(f'Test dataset generated with {test_data.__len__()} samples.')

    # generate the validation data (if needed)
    if len(program_num_split) == 3:
        validation_data = raw_data.split_by_program_num(program_num_split[2])
        logging.info(f'Validation dataset generated with {validation_data.__len__()} samples.')


    # Step 4. Normalized the training data.
    


    # Step 5. Export the data sets and normalization parameters. 



# %%
if __name__ == "__main__":
    main()