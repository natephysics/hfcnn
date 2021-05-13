from hfcnn.lib import yaml_tools

class NetworkConfig():
    def __init__(self, path_to_config: str):
        self.config = yaml_tools.import_configuration(path_to_config)

    
    def export(self, path_to_export):
        yaml_tools.export_configuration(path_to_export, self.config)
