# %%
from hfcnn import config, yaml_tools
import rna
import re
from typing import Iterator, Union

options = config.construct_options_dict()

REGEX = r"[^${\}]+(?=})"

def generate_dvc_pipeline():
    # check to see if the path is defined in 
    # first check the path specified in the config (see if statement)


    # import the yaml template
    dvc_template = yaml_tools.import_configuration(options['dvc_template_path'])

    # filter the paths in yaml template with the config paths
    variable_replace(dvc_template)

    # copy the template to the dvc_pipeline_path
    rna.path.cp(options['dvc_template_path'], options['dvc_pipeline_path'])
    
    # export the filtered dvc template
    yaml_tools.export_configuration(options['dvc_pipeline_path'], dvc_template)

def variable_replace(dvc_template: Union[dict, list]):
    """variable_replace takes a DVC template file and replaces all variables in the template
    with the paths from the config. 

    Args:
        dvc_template ([dict, list]): the imported yaml dict or list
    """
    # Because yamls can contain dicts and lists we employ a fixable structure for recursive calls.
    # if the input is a dict use items()
    if isinstance(dvc_template, dict):
        value_pairs = dvc_template.items()
    # if the input is a list, use enumerate()
    elif isinstance(dvc_template, list):
        value_pairs = enumerate(dvc_template)

    for key, key_path in value_pairs:
        # if the key_path is a dictionary, run recursively. 
        if isinstance(key_path, dict):
            variable_replace(key_path)

        # if the key_path is a list, run recursively. 
        elif isinstance(key_path, list):
            variable_replace(key_path)

        # if the key_path is a string, pattern match and replace.
        elif isinstance(key_path, str):
            matches = re.finditer(REGEX, key_path)
            for match in list(matches):
                start, end = match.span()
                # recursively replace the match by values from options
                if match.group() in options.keys():
                    dvc_template[key] = (
                        key_path[: start - 2]
                        + options[match.group()]
                        + key_path[end + 1 :]
                    )
                else:
                    raise ValueError(f'The path {match.group()} was not found in the config.')
    return 



generate_dvc_pipeline()
