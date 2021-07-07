#!/usr/bin/env  # pylint: disable=missing-class-docstring,too-few-public-methods,abstract-method
"""
"""
# %%
import os
import pathlib
import configparser
import re
import rna

# Cache of already loaded .cfg files
CONFIG_CACHE = {}  

# This is a memory on options set by the user with set_option
# These options override the options in CONFIG_CACHE
DYNAMIC_OPTIONS = {}  

REGEX = r"[^${\}]+(?=})"


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


VARIABLES = dict(
    HOME = str(pathlib.Path.home()),
    HFCNN_PACKAGE = os.path.dirname(__file__),
)
VARIABLES['PROJECT_ROOT'] = os.path.dirname(VARIABLES['HFCNN_PACKAGE'])


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
        rna.path.resolve(VARIABLES['PROJECT_ROOT'], "config", "config.cfg"),
        rna.path.resolve(VARIABLES['HFCNN_PACKAGE'], "default.cfg"),
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
        # replace explicit mentions of VARIABLES
        for env_var, env_value in VARIABLES.items():
            val = val.replace("$" + env_var, env_value)

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
    # impliments an inheritance between a parent and child class
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
        'pp_log_path',
        'nc_log_path',
        'train_log_path',
        'model_path'
        ]
    options = {}
    # import the paths into the options dictonary
    for path in paths_list:
        # grab the option from the .cfg file
        option = get_option("paths", path)
        # verify the option is set in the config file
        if option != None:
            options[path] = option

    return options
# %%
