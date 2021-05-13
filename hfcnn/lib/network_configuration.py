from hfcnn.lib import yaml_tools

# def config_tests(config: dict):
#     """A series of tests to assure the config file contains the relevant
#     parameters. 

#     Args:
#         config (dict): the dictionary containing the relevant parameters. 
#     """
#     # Check for file path
#     expected_keys = ['raw_data_path',
#     'raw_df_path',
#     'processed_data_path',
#     'filters_to_apply'
#     ]

class GenerateConfig():
    def __init__(self, path_to_config: str):
        self.config = yaml_tools.import_configuration(path_to_config)

    def export(self, path_to_export):
        yaml_tools.export_configuration(path_to_export, self.config)

    def value(self, key):
        return self.config(key)

    def keys(self):
        return self.config.keys()

        