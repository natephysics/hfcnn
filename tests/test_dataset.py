import unittest
import numpy as np
from numpy.testing._private.utils import assert_equal
from hfcnn.lib.dataset import HeatLoadDataset
from hfcnn.lib import files

class TestDataSetClass(unittest.TestCase):
    """Tests for data set classes."""
    def setUp(self):
        # test dataframe
        self.df = 'tests/resources/test_df.hkl'

        # actual heat load data
        self.data_good = files.import_file_from_local_cache('tests/resources/good.hkl')
        self.data_bad = files.import_file_from_local_cache('tests/resources/bad.hkl')
        self.PC1 = np.array([8698.])

    def test_HeatLoadDataset(self):

        """Testing the HeatLoadDataset class."""
        # Check __init__ method for both file import and manual import
        dataset_file = HeatLoadDataset(self.df, 'tests')
        dataset_df = HeatLoadDataset(files.import_file_from_local_cache('tests/resources/test_df.hkl'), 'tests')

        # Check __len__ method
        assert_equal(dataset_file.__len__(), 2)
        assert_equal(dataset_df.__len__(), 2)

        # check __getitem__ method
        good = dataset_file.__getitem__('good')
        bad = dataset_df.__getitem__('bad')
        assert_equal(good['image'], self.data_good)
        assert_equal(good['label'], self.PC1)
        assert_equal(bad['image'], self.data_bad)
        assert_equal(bad['label'], self.PC1)




