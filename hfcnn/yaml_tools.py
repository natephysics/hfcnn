from ruamel.yaml import YAML
from hfcnn import config

def import_configuration(params_path: str):
    """Imports the processing and training parameter yaml file.

    Args:
        config_path (str): path to processing and training parameter yaml file.
    """
    yaml = YAML()

    # load the file
    with open(params_path, 'r') as files:
        model_params = yaml.load(files)

    # list of types to specify    
    model_params_type = [
        ('batch_size', int),
        ('val_evaluation_size', int)
    ]

    # enforce type on specific parameters
    for param, param_type in model_params_type:
        if param in model_params.keys():
            model_params[param] = param_type(model_params[param])

    return model_params

def export_configuration(config_path: str, network_config: dict):
    """Exports a configuration file to a yaml document.

    Args:
        config_path (str): Path to yaml file.
        # network_config (dict): Configuration dictionary. 
    """
    yaml = YAML()
    with open(config_path, 'w') as file:
        yaml.dump(network_config, file)