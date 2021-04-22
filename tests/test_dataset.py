import unittest
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

    def test_HeatLoadDataset(self):
        """Testing the HeatLoadDataset class."""
        dataset = HeatLoadDataset(self.df, 'tests')

        # Check __len__ method
        assert_equal(dataset.__len__(), 2)

        # check __getitem__ method
        bad = dataset.__getitem__('bad')
        assert_equal(bad['image'], self.data_bad)

        # check __getitem__ method
        good = dataset.__getitem__('good')
        assert_equal(good['image'], self.data_good)