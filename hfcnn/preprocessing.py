from hfcnn import dataset, filters, yaml_tools, config
from hfcnn.utils import get_logger, instantiate_list
from omegaconf import DictConfig
import os

default_paths = config.build_default_paths()

def prepare_data(cfg: DictConfig, **kwargs) -> None:

    log = get_logger(__name__)

    ##################
    # Importing Data #
    ##################

    # Adjust paths to the original working directory. 
    data_root = os.path.join(cfg.orig_wd, cfg.data_folder)
    raw_data_folder = os.path.join(data_root, default_paths['raw_folder'])
    test_data_path = os.path.join(data_root, default_paths['test'])
    train_data_path = os.path.join(data_root, default_paths['train'])
    val_data_path = os.path.join(data_root, default_paths['validation'])

    data_settings = {
        'img_dir': raw_data_folder,
        'label_list': cfg.label_names
    }

    raw_data = dataset.HeatLoadDataset(
        os.path.join(raw_data_folder, cfg.data_file), 
        **data_settings
        )
    
    # If there's test data, exclude it from the data set
    if os.path.exists(test_data_path):
            test_data = dataset.HeatLoadDataset(
                # Path to processed dataframe
                test_data_path,
                # Path to the raw image files
                img_dir = raw_data_folder
                )

            # We want to exclude any program numbers from the test set.
            test_data_program_nums = test_data.program_nums()
            raw_data_program_nums = raw_data.program_nums()

            # we want the list of programs in raw not contained in test
            diff_program_nums = list(set(raw_data_program_nums).difference(test_data_program_nums))
            
            # update raw
            raw_data = raw_data.split_by_program_num(diff_program_nums)
    

    ## TODO: Move this to the import step
    # Adding new columns for IA
    raw_data.img_labels['IA'] = raw_data.img_labels['PC1'].div(raw_data.img_labels['NPC1'])


    step_1_message = f"Preprocseeing Data: Imported {raw_data.__len__()} images from the raw data set."
    log.info(step_1_message)
    print(step_1_message)


    ########################################
    #### Applying filter(s) to the data ####
    ########################################
    # Filtering which data is selected to be used by the CNN
    # Import data selection filters 

    list_of_filters = instantiate_list(cfg.filters)

    if len(list_of_filters) == 0:
        log.info("Preprocseeing Data: No filters to apply. Skipping step")
    else:
        print(len(list_of_filters))
        print('Preprocseeing Data: Begining to filter data, for larger data sets this may take a while')
        raw_data = raw_data.apply(list_of_filters)
        step_2_message = f"{raw_data.__len__()} images remain after applying {len(list_of_filters)} filters"
        log.info(step_2_message)
        print(step_2_message)


    #########################
    #### Train/Val Split ####
    #########################
    # Because data for a given program number is likely corolated we'll divide
    # the sets up by program number.


    training_data, validation_data = raw_data.validation_split(cfg.val_split)

    log.info(f"Preprocseeing Data: Training dataset generated with {training_data.__len__()} samples./n \
    Preprocseeing Data: Validation dataset generated with {validation_data.__len__()} samples.")


    ##################################################
    #### Normalized the training data and labels. ####
    ##################################################

    training_data.normalize_data()
    step_4_message = f"Preprocseeing Data: Training set standardization parameters.\n \
          mean: {training_data.settings['norm_param']['image_labels'][0]}, \n \
          std: {training_data.settings['norm_param']['image_labels'][1]}."
    log.info(step_4_message)
    print(step_4_message)

    if cfg.normalize_labels == True:
        training_data.normalize_labels(cfg.label_names)

    # Use the same settings for the validation and save the files
    data_settings = training_data.settings

    validation_data.import_settings(data_settings)
    validation_data.to_file(val_data_path)

    # if test data is available, update the test data as well
    if os.path.exists(test_data_path):
        test_data.import_settings(data_settings)
        test_data.to_file(test_data_path)

    training_data.to_file(train_data_path)