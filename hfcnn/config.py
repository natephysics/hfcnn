import os
import configparser
import re
import rna

# Cache of already loaded .cfg files
CONFIG_CACHE = {}  

# This is a memory on options set by the user with set_option
# These options override the options in CONFIG_CACHE
DYNAMIC_OPTIONS = {}  

REGEX = r"[^${\}]+(?=})"

package_path = os.path.basename(os.path.dirname(__file__))

def get_config(config_path) -> configparser.ConfigParser:
    """
        Retrieve a .cfg config file
    """
    if config_path not in CONFIG_CACHE:
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            CONFIG_CACHE[config_path] = config
        else:
            CONFIG_CACHE[config_path] = None
    return CONFIG_CACHE[config_path]

def get_option(section: str, option: str, fallback=False):
    """
    Retrieve option from section.
    """
    # First check DYNAMIC_OPTIONS for option, if found return (ignoring .cfg file)
    if section in DYNAMIC_OPTIONS and option in DYNAMIC_OPTIONS[section]:
        # set_option was active
        return DYNAMIC_OPTIONS[section][option]
    
    # Sets the order of the paths to check for a config file
    config_resolve_order = [
        rna.path.resolve("config", "config.cfg"),
        rna.path.resolve(package_path, "default_settings", "default_config.cfg"),
    ]
    val = None
    for config_path in config_resolve_order:
        config = get_config(config_path)
        if config is None:
            continue
        try:
            val = config.get(section, option)
        except (configparser.NoOptionError, configparser.NoSectionError):
            val = None

        if val is not None:
            break

    if val is not None:
        # replace by regex with recursive get_option
        # example:
        ## [global]
        ## data = data/path
        ##
        ## [paths]
        ## raw_data_path = ${global.data}/raw/ 
        #
        matches = re.finditer(REGEX, val)
        for match in reversed(list(matches)):
            start, end = match.span()
            # recursively replace the match by get_option
            val = (
                val[: start - 2]
                + get_option(*match.group().rsplit(".", 1))
                + val[end + 1 :]
            )
    # implements an inheritance between a parent and child class
    # example:
    ## [parent]
    ## att1 = 3
    ##
    ## [parent.child]
    ## att2 = b
    #
    # parent.child will have both attributes, parent will only have att1.
    elif val is None and fallback:
        val = get_option(section.rpartition(".")[0], option, fallback=fallback)

    return val


def set_option(section: str, option: str, value: any):
    """Sets an option in the DYNAMIC_OPTIONS which will override
    options loaded from .cfg"""
    if section not in DYNAMIC_OPTIONS:
        DYNAMIC_OPTIONS[section] = {}
    DYNAMIC_OPTIONS[section][option] = value


def unset_option(section: str, option: str):
    """Removes an option already set in the DYNAMIC_OPTIONS."""
    del DYNAMIC_OPTIONS[section][option]

def construct_options_dict():
    """Creates a diction of the options available in the config file.
    """
    # List of paths to parse .cfg file for
    paths_list = [
        'raw_data_path', 
        'raw_df_path', 
        'processed_data_path', 
        'test_df_path',
        'train_df_path',
        'validation_df_path',
        'log_path',
        'untrained_model_path',
        'training_model_path',
        'best_model_path',
        'tensorboard_dir',
        'data_params_path',
        'training_results',
        'dvc_pipeline_path',
        ]
    options = {}
    # import the paths into the options dictionary
    for path in paths_list:
        # grab the option from the .cfg file
        option = get_option("paths", path)
        # verify the option is set in the config file
        if option != None:
            options[path] = option

    # Add links to prespecified paths for configuration files (which include resolution orders)
    # path_list = (key, filename)
    path_list = [
        ('dvc_template_path', 'dvc.template.yaml'),
        ('preprocessing_config_path', 'preprocessing.yaml'),
        ('network_config_path', 'network_construction.yaml'),
        ('training_config_path', 'training.yaml')
    ]

    for key, filename in path_list:
        options[key] = resolve_path(filename)

    return options

def resolve_path(filename: str):
    """Takes a filename and checks to see which paths it exists on in the resolve order.

    Args:
        filename (str): filename to check for.

    Raises:
        ValueError: Can't find an valid file for any possible paths.

    Returns:
        resolved_path (str): Returns the correct path.
    """
    # list of possible paths
    template_resolve_order = [
        os.path.join(get_option("global", "config"), filename).replace("\\","/"), # check the config folder
        os.path.join(package_path, "default_settings", "default_" + filename).replace("\\","/") # otherwise use default
    ]

    # check to see if the file exists on those paths
    template_path = None
    for possible_dvc_template in template_resolve_order:
        if os.path.exists(possible_dvc_template):
            template_path = possible_dvc_template
            break
    # if no valid paths, return error
    if template_path == None:
        raise ValueError(f'Could not resolve working path to {filename}')
    return template_path

construct_options_dict()