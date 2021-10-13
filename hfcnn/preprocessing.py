from hfcnn import dataset, filters, yaml_tools, config
from hfcnn.utils import get_logger
from omegaconf import DictConfig
import os


def prepare_data(cfg: DictConfig, **kwargs) -> None:

    log = get_logger(__name__)

    ##################
    # Importing Data #
    ##################

    raw_data_path = os.path.join(cfg.orig_cwd, cfg.raw_data_path)
    processed_data_path = os.path.join(cfg.orig_cwd, cfg.processed_path)

    data_settings = {
        'img_dir': os.path.dirname(raw_data_path),
        'label_list': cfg.label_names
    }

    raw_data = dataset.HeatLoadDataset(raw_data_path, **data_settings)
    
    if os.path.exists(os.path.join(processed_data_path, 'processed/'))

    # Adding new columns for IA
    raw_data.img_labels['IA'] = raw_data.img_labels['PC1'].div(raw_data.img_labels['NPC1'])

    step_1_message = f"Preprocseeing Data: Imported {raw_data.__len__()} images from the raw data set."
    log.info(step_1_message)
    print(step_1_message)