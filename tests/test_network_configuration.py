"""Tests for filters.py."""
import unittest
import numpy as np
from hfcnn import files, network_configuration
from numpy.testing._private.utils import assert_equal

## TODO fix test structures to reflect test_dataset

class TestGenerateConfigClass(unittest.TestCase):
    """Tests for data set classes."""
    def setUp(self):
        # test dataframe
        self.genconfig = network_configuration.GenerateConfig('tests/resources/preprossing_data.yaml')


    def test_keys(self):
        assert_equal(self.genconfig.get('processed_data_path'), 'data/processed/')