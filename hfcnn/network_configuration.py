from hickle.hickle_legacy import NoneType
from hfcnn import yaml_tools

class GenerateConfig():
    def __init__(self, path_to_config: str):
        self.config = yaml_tools.import_configuration(path_to_config)

    def export(self, path_to_export):
        yaml_tools.export_configuration(path_to_export, self.config)

    def get(self, key):
        return self.config.get(key)

    def keys(self):
        return self.config.keys()

    def num_of_filters(self):
        if self.get('filters_to_apply') == None:
            return 0
        else: 
            return len(self.get('filters_to_apply'))
