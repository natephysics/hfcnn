from hfcnn import dataset, utils
from hfcnn.utils import get_logger, instantiate_list
from omegaconf import DictConfig
import os

def prepare_test_data(cfg: DictConfig, **kwargs) -> None:

    default_paths = utils.build_default_paths(cfg)

    log = get_logger(__name__)

    ##################
    # Importing Data #
    ##################

    data_settings = {
        'img_dir': default_paths['raw_folder'],
        'label_list': cfg.label_names
    }

    raw_data = dataset.HeatLoadDataset(
        os.path.join(default_paths['raw_folder'], cfg.data_file), 
        **data_settings
        )
    

    ## TODO: Move this to the import step
    # Adding new columns for IA
    raw_data.img_labels['IA'] = raw_data.img_labels['PC1'].div(raw_data.img_labels['NPC1'])

    log.info(f"Preprocseeing Data: Imported {raw_data.__len__()} images from the raw data set.")

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
        log.info(f"{raw_data.__len__()} images remain after applying {len(list_of_filters)} filters")

    ####################
    #### Test Split ####
    ####################
    # Because data for a given program number is likely corolated we'll divide
    # the sets up by program number.

    _, test_data = raw_data.validation_split(cfg.test_split)

    log.info(f"Preprocseeing Data: Test dataset generated with {validation_data.__len__()} samples.")

    # Save the data
    test_data.to_file(default_paths['test'])

    log.info("Test set gereration Complete.")


def prepare_data(cfg: DictConfig, **kwargs) -> None:

    default_paths = utils.build_default_paths(cfg)

    log = get_logger(__name__)

    ##################
    # Importing Data #
    ##################

    data_settings = {
        'img_dir': default_paths['raw_folder'],
        'label_list': cfg.label_names
    }

    raw_data = dataset.HeatLoadDataset(
        os.path.join(default_paths['raw_folder'], cfg.data_file), 
        **data_settings
        )
    
    # If there's test data, exclude it from the data set
    if os.path.exists(default_paths['test']):
            test_data = dataset.HeatLoadDataset(
                # Path to processed dataframe
                default_paths['test'],
                # Path to the raw image files
                img_dir = default_paths['raw_folder']
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

    log.info(f"Preprocseeing Data: Imported {raw_data.__len__()} images from the raw data set.")

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
        log.info(f"{raw_data.__len__()} images remain after applying {len(list_of_filters)} filters")

    #########################
    #### Train/Val Split ####
    #########################
    # Because data for a given program number is likely corolated we'll divide
    # the sets up by program number.

    training_data, validation_data = raw_data.validation_split(cfg.val_split)

    log.info(f"Preprocseeing Data: Training dataset generated with {training_data.__len__()} samples.")
    log.info(f"Preprocseeing Data: Validation dataset generated with {validation_data.__len__()} samples.")


    ##################################################
    #### Normalized the training data and labels. ####
    ##################################################

    training_data.normalize_data()
    log.info(f"Preprocseeing Data: Training set standardization parameters.")
    log.info(f"mean: {training_data.settings['norm_param']['image_labels'][0]}")
    log.info(f"std: {training_data.settings['norm_param']['image_labels'][1]}")

    

    if cfg.normalize_labels == True:
        training_data.normalize_labels(cfg.label_names)
        data_settings = training_data.settings
        
        # update valdation
        validation_data.import_settings(data_settings)
        validation_data.to_file(default_paths['validation'])

    # if test data is available, update the test data as well
    if os.path.exists(default_paths['test']):
        test_data.import_settings(data_settings)
        test_data.to_file(default_paths['test'])

    training_data.to_file(default_paths['train'])

    log.info("Preprocessing Complete.")