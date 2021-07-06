#!/usr/bin/env  # pylint: disable=missing-class-docstring,too-few-public-methods,abstract-method
"""
"""
# %%
import os
import pathlib
import typing
import configparser
import re
import numpy as np
import rna
import hfcnn



CONFIG_CACHE = {}  # Cache of already loaded .cfg files
DYNAMIC_OPTIONS = {}  # This is a memory on options set by the user with set_option


def get_project_root() -> pathlib.Path:
    """Returns project root directory"""
    return

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

REGEX = r"[^${\}]+(?=})"

def get_option(section: str, option: str, fallback=False):
    """
    Retrieve option from section.
    """
    ### Ask about this ###
    if section in DYNAMIC_OPTIONS and option in DYNAMIC_OPTIONS[section]:
        # set_option was active
        return DYNAMIC_OPTIONS[section][option]

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
        matches = re.finditer(REGEX, val)
        for match in reversed(list(matches)):
            start, end = match.span()
            # recursively replace the match by get_option
            val = (
                val[: start - 2]
                + get_option(*match.group().rsplit(".", 1))
                + val[end + 1 :]
            )
    elif val is None and fallback:
        val = get_option(section.rpartition(".")[0], option, fallback=fallback)

    return val


def set_option(section: str, option: str, value: any):
    if section not in DYNAMIC_OPTIONS:
        DYNAMIC_OPTIONS[section] = {}
    DYNAMIC_OPTIONS[section][option] = value


def unset_option(section: str, option: str):
    del DYNAMIC_OPTIONS[section][option]

def get_mgrid_options(return_type=None):
    """
    Reads the [mgrid] options and returns base vectors and iter_order (see MGrid or
    tfields.TensorGrid)

    Args:
        return_type: if not None, cast directly to type
    """
    nfp = int(get_option("mgrid", "nfp"))
    base_vectors = []
    iter_order = []
    for var in ["r", "phi", "z"]:
        xmin = get_option("mgrid." + var, "min")
        xmin = float(xmin) if xmin != "None" else 0
        xmax = get_option("mgrid." + var, "max")
        xmax = float(xmax) if xmax != "None" else 2 * np.pi / nfp
        num = complex(get_option("mgrid." + var, "num") + "j")
        base_vectors.append((xmin, xmax, num))
        iter_order.append(int(get_option("mgrid." + var, "iter")))

    return base_vectors