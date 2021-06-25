# %%
import json
from collections import OrderedDict

def export_nn_structure(path, nn_structure):
    with open(path, 'w') as file:
        json_string = json.dumps(
            nn_structure, 
            default=lambda o: o.__dict__, 
            sort_keys=True, 
            indent=2,
            )
        file.write(json_string)

def import_nn_structure(path):
    with open(path) as f:
        data = json.load(f)
    return data
# %%
