import argparse
from hfcnn import dataset, filters, yaml_tools, config
import os
import logging

# import the options
options = config.construct_options_dict()

logging.basicConfig(
    filename=options['pp_log_path'],
    filemode="a",
    force=True,
    format="%(asctime)s %(msecs)d- %(process)d -%(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S %p",
    level=logging.DEBUG,
)

def main():
    # grab the path from the argument
    parser = argparse.ArgumentParser()
    # Test/Train/Validation Split Ratio
    # For 80/20 Train/Test, just return --train_ratio 80.
    # The remainder of the data set is assumed to be test.
    #
    # For 70/15/15 Train/Test/Validate, return --train_ratio 70, --test_ratio 15
    # The remainder of the data set is assumed to be validate. 
    parser.add_argument("--train_ratio", help="the ratio of the data set to dedicate to training", type=int, required=True)
    parser.add_argument("--test_ratio", help="only specify if plan to train/test/validate split", type=int)
    parser.add_argument("--filter_list_path", help="Path to yaml containing the filters to be applied")
    
    train_ratio = parser.parse_args().train_ratio
    # extract the train/test/validation split
    train_test_split = [train_ratio/100]
    if parser.parse_args().test_ratio:
        train_test_split.append(parser.parse_args().test_ratio/100)

    # Import data selection filters 
    list_of_filters = []

    if parser.parse_args().filter_list_path:
        filter_list_path = parser.parse_args().filter_list_path
        list_of_filters = yaml_tools.import_configuration(filter_list_path)
        if list_of_filters == None:
            list_of_filters = []

    #### Step 1. Import the raw the data ####
    raw_data = dataset.HeatLoadDataset(options["raw_df_path"], options["raw_data_path"])
    logging.info(f"Imported {raw_data.__len__()} images from the raw data set")
    print(f"Imported {raw_data.__len__()} images from the raw data set")

    #### Step 2. Allpying filter(s) to the data ####
    if len(list_of_filters) == 0:
        logging.info("No filters to apply. Skipping step")
    else:
        for filter in list_of_filters:
            logging.info(f"Applying {filter[0]} filters")
            raw_data = raw_data.apply(filters.return_filter(*filter))
            logging.info(
                f"{raw_data.__len__()} images remain after applying {filter[0]} filter"
            )
            print(
                f"{raw_data.__len__()} images remain after applying {filter[0]} filter"
            )

    #### Step 3. Test/Train/Val Split ####
    # Because data for a given program number is likely corolated we'll divide
    # the sets up by program number.

    # generate lists of program numbers for each train/test/val split
    program_num_split = filters.split(
        raw_data.program_nums(), train_test_split
    )

    # generate the training data
    training_data = raw_data.split_by_program_num(program_num_split[0])
    logging.info(f"Training dataset generated with {training_data.__len__()} samples.")

    # generate the test data
    test_data = raw_data.split_by_program_num(program_num_split[1])
    logging.info(f"Test dataset generated with {test_data.__len__()} samples.")

    # generate the validation data (if needed)
    if len(program_num_split) == 3:
        validation_data = raw_data.split_by_program_num(program_num_split[2])
        logging.info(
            f"Validation dataset generated with {validation_data.__len__()} samples."
        )

    #### Step 4. Normalized the training data. ####
    training_data.normalize()
    logging.info(
        f"Training set standardization parameters. mean: {training_data.mean}, std: {training_data.std}."
    )
    print(
        str(
            f"Training set standardization parameters. mean: {training_data.mean}, std: {training_data.std}."
        )
    )

    #### Step 5. Export the data sets and standardization parameters. ####
    training_data.to_file(options["train_df_path"])
    test_data.to_file(options["test_df_path"])
    if len(program_num_split) == 3:
        validation_data.to_file(options["validation_df_path"])
    else:
        if os.path.isfile(options["validation_df_path"]):
            os.remove(options["validation_df_path"])
    logging.info(f"Datasets exported to disk.")

if __name__ == "__main__":
    main()