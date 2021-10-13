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

cwd = os.path.basename(os.path.dirname(__file__))

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

def get_option(section: str, option: str, owd: str=cwd, fallback=False):
    """
    Retrieve option from section.
    """
    # First check DYNAMIC_OPTIONS for option, if found return (ignoring .cfg file)
    if section in DYNAMIC_OPTIONS and option in DYNAMIC_OPTIONS[section]:
        # set_option was active
        return DYNAMIC_OPTIONS[section][option]
    
    # Sets the order of the paths to check for a config file
    config_resolve_order = [
        rna.path.resolve(owd, "configs", "config.cfg"),
        rna.path.resolve(owd, "default_settings", "default_config.cfg"),
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
                + get_option(*match.group().rsplit(".", 1), owd=owd)
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
        val = get_option(section.rpartition(".")[0], option, owd=owd, fallback=fallback)

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

def build_default_paths(owd: str=None):
    """Builds a dictionary of default paths. 

    Args:
        owd (str): the working directory that the path is relative to. 

    Returns:
        [dict]: Dict with the default paths. 
    """
    if owd is None:
        # check for an original working directory
        if 'OWD' in os.environ.keys():
            owd = os.environ['OWD']
        else:
            owd = os.getcwd()

    # List of paths to parse .cfg file for
    paths_list = [
        'dvc_pipeline_path',
        'train',
        'validation',
        'test',
        'raw_folder'
        ]
    options = {}
    # import the paths into the options dictionary
    for path in paths_list:
        # grab the option from the .cfg file
        option = get_option("paths", path, owd=owd)
        # verify the option is set in the config file
        if option != None:
            options[path] = option

    # Add links to prespecified paths for configuration files
    # path_list = (key, filename)
    path_list = [
        ('dvc_template_path', 'dvc.template.yaml')
    ]

    for key, filename in path_list:
        options[key] = resolve_path(filename, owd=owd)

    return options

def resolve_path(filename: str, owd: str=cwd):
    """Takes a filename and checks to see which paths it exists on in the resolve order.

    Args:
        filename (str): filename to check for.
        owd (str): the working directory that the path is relative to. 

    Raises:
        ValueError: Can't find an valid file for any possible paths.

    Returns:
        resolved_path (str): Returns the correct path.
    """
    # list of possible paths
    print(get_option("global", "config", owd=owd))
    template_resolve_order = [
        os.path.join(owd, get_option("global", "config", owd=owd), filename).replace("\\","/"), # check the config folder
        os.path.join(owd, "default_settings", "default_" + filename).replace("\\","/") # otherwise use default
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