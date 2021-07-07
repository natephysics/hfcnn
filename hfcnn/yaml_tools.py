# %%
from ruamel.yaml import YAML

def import_configuration(config_path: str):
    """Imports the processing and training parameter yaml file.

    Args:
        config_path (str): path to processing and training parameter yaml file.
    """
    yaml = YAML()

    with open(config_path, 'r') as files:
        config_dict = yaml.load(files)
    return config_dict

def export_configuration(config_path: str, network_config: dict):
    """Exports a configuration file to a yaml document.

    Args:
        config_path (str): Path to yaml file.
        # network_config (dict): Configuration dictionary. 
    """
    yaml = YAML()
    with open(config_path, 'w') as file:
        yaml.dump(network_config, file)