from hfcnn import dataset, filters, yaml_tools, config
import logging

## TODO: Break out test generation  

# import the options
options = config.construct_options_dict()

logging.basicConfig(
    filename=options['log_path'],
    filemode="a",
    force=True,
    format="%(asctime)s %(msecs)d- %(process)d -%(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S %p",
    level=logging.DEBUG,
)

data_params = yaml_tools.import_configuration(options['preprocessing_config_path'])

label_list = data_params['label_list']

def main():
    #### Step 1. Import the raw the data ####
    data_settings = {
        'img_dir': options["raw_data_path"],
        'label_list': label_list
        }
    raw_data = dataset.HeatLoadDataset(options["raw_df_path"], **data_settings)
    step_1_message = f"Preprocseeing Data: Imported {raw_data.__len__()} images from the raw data set"

    # If a test dataset is already available, import this as well
    if data_params['exclude_test']:
        test_data = dataset.HeatLoadDataset(options["test_df_path"])
        test_program_nums = test_data.program_nums().tolist()
        raw_data_program_nums = raw_data.program_nums().tolist()
        for test_num in test_program_nums:
            if test_num in raw_data_program_nums:
                raw_data_program_nums.remove(test_num)
        raw_data = raw_data.split_by_program_num(raw_data_program_nums)
    
    # Adding new columns for IA
    raw_data.img_labels['IA'] = raw_data.img_labels['PC1'].div(raw_data.img_labels['NPC1'])

    
    logging.info(step_1_message)
    print(step_1_message)

    #### Step 2. Applying filter(s) to the data ####
    # Filtering which data is selected to be used by the CNN
    # Import data selection filters 
    list_of_filters = data_params['filter_list']

    if len(list_of_filters) == 0:
        logging.info("Preprocseeing Data: No filters to apply. Skipping step")
    else:
        for filter in list_of_filters:
            print('Preprocseeing Data: Begining to filter data, for larger data sets this may take a while')
            raw_data = raw_data.apply(filters.return_filter(*filter))
            step_2_message = f"{raw_data.__len__()} images remain after applying {filter[0]} filter"
            logging.info(step_2_message)
            print(step_2_message)

    #### Step 3. Test/Train/Val Split ####
    # Because data for a given program number is likely corolated we'll divide
    # the sets up by program number.

    # generate lists of program numbers for each train/test/val split
    program_num_split = filters.split(
        raw_data.program_nums(), data_params['train_ratio']
    )

    # generate the training data
    training_data = raw_data.split_by_program_num(program_num_split[0])
    logging.info(f"Preprocseeing Data: Training dataset generated with {training_data.__len__()} samples.")

    # generate the test data
    validation_data = raw_data.split_by_program_num(program_num_split[1])
    logging.info(f"Preprocseeing Data: Validation dataset generated with {validation_data.__len__()} samples.")

    # generate the validation data (if needed)
    if len(program_num_split) == 3:
        test_data = raw_data.split_by_program_num(program_num_split[2])
        logging.info(
            f"Preprocseeing Data: Test dataset generated with {test_data.__len__()} samples."
        )

    #### Step 4. Normalized the training data and labels. ####
    training_data.normalize_data()
    step_4_message = f"Preprocseeing Data: Training set standardization parameters. mean: {training_data.settings['norm_param']['image_labels'][0]}, \
        std: {training_data.settings['norm_param']['image_labels'][1]}."
    logging.info(step_4_message)
    print(step_4_message)

    training_data.normalize_labels(label_list)

    #### Step 5. Export the data sets and standardization parameters. ####

    training_data.to_file(options["train_df_path"])

    # Use the same settings for the validation and test data set
    data_settings = training_data.settings

    validation_data.import_settings(data_settings)
    validation_data.to_file(options["validation_df_path"])
    
    if (len(program_num_split) == 3) or (data_params['exclude_test']):
        test_data.import_settings(data_settings)
        test_data.to_file(options["test_df_path"])
    logging.info(f"Preprocseeing Data: Datasets exported to disk.")

if __name__ == "__main__":
    main()