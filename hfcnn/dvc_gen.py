# %%
from hfcnn import config, yaml_tools
import pathlib
import rna
import os
import re

options = config.construct_options_dict()

VARIABLES = dict(
    HOME = str(pathlib.Path.home()),
    HFCNN_PACKAGE = os.path.dirname(__file__),
)
VARIABLES['PROJECT_ROOT'] = os.path.dirname(VARIABLES['HFCNN_PACKAGE'])

REGEX = r"[^${\}]+(?=})"

def generate_dvc_pipeline():
    # check to see if the path is defined in 
    # first check the path specified in the config (see if statement)
    dvc_template_resolve_order = [
        rna.path.resolve(VARIABLES['PROJECT_ROOT'], "config", "dvc.template.yaml"), # check the config folder
        rna.path.resolve(VARIABLES['HFCNN_PACKAGE'], "default_settings", "dvc.template.yaml") # otherwise use default
    ]
    if 'dvc_template_path' in options.keys():
        # first check the path specified in the config 
        dvc_template_path = options['dvc_template_path']
    else:
        # if not, check other paths
        dvc_template_path = None
        for possible_dvc_template in dvc_template_resolve_order:
            if os.path.exists(possible_dvc_template):
                dvc_template_path = possible_dvc_template
                break
                
        if dvc_template_path == None:
            raise ValueError('Could not resolve working path to dvc.template.yaml')

    # import the yaml template
    dvc_template = yaml_tools.import_configuration(dvc_template_path)

    # filter the paths in yaml template with the config paths
    variable_replace(dvc_template)

    # copy the template to the dvc_pipeline_path
    rna.path.cp(dvc_template_path, options['dvc_pipeline_path'])
    
    # export the filtered dvc template
    yaml_tools.export_configuration(options['dvc_pipeline_path'], dvc_template)

def variable_replace(dvc_template):
    """variable_replace takes a DVC template file and replaces all variables in the template
    with the paths from the config. 

    Args:
        dvc_template ([type]): [description]
    """
    for key, key_path in dvc_template.items():
        if isinstance(key_path, dict):
            variable_replace(key_path)
        else:
            matches = re.finditer(REGEX, key_path)
            for match in list(matches):
                start, end = match.span()
                # recursively replace the match by values from options
                if match.group() in options.keys():
                    dvc_template[key] = options[match.group()]
                else:
                    raise ValueError(f'The path {match.group()} was not found in the config.')
    return 

generate_dvc_pipeline()



# %%
