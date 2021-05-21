import unittest
import numpy as np
from numpy.testing._private.utils import assert_equal
from hfcnn.lib.dataset import HeatLoadDataset
from hfcnn.lib import files, filters

class TestDataSetClass(unittest.TestCase):
    """Tests for data set classes."""
    def setUp(self):
        # test dataframe
        self.df = 'tests/resources/test_df.hkl'

        # actual heat load data
        self.data_good = files.import_file_from_local_cache('tests/resources/good.hkl')
        self.data_bad = files.import_file_from_local_cache('tests/resources/bad.hkl')
        self.PC1 = np.array([8698.])
        self.index_bad = 0
        self.index_good = 1

    def test_HeatLoadDataset(self):

        """Testing the HeatLoadDataset class."""
        # Check __init__ method for both file import and manual import
        dataset_file = HeatLoadDataset(self.df, 'tests')
        dataset_df = HeatLoadDataset(
            files.import_file_from_local_cache('tests/resources/test_df.hkl'), 
            'tests'
            )

        # Check __len__ method  
        assert_equal(dataset_file.__len__(), 2)
        assert_equal(dataset_df.__len__(), 2)

        # check __getitem__ method
        good = dataset_file.__getitem__(self.index_good)
        bad = dataset_df.__getitem__(self.index_bad)
        assert_equal(good['image'], self.data_good)
        assert_equal(good['label'], self.PC1)
        assert_equal(bad['image'], self.data_bad)
        assert_equal(bad['label'], self.PC1)

        # check apply method
        dataset_good = dataset_df.apply(
            filters.return_filter(*["data_selection", dataset_df.img_dir])
            )

        assert_equal(dataset_good.__len__(), 1)


        # Check split_by_program_num method
        prog_num_list1 = ['20180829.36']
        prog_num_list2 = ['20180829.32', '20180829.36']

        len1 = dataset_df.split_by_program_num(prog_num_list1).__len__()
        len2 = dataset_df.split_by_program_num(prog_num_list2).__len__()

        assert_equal(len1, 1)
        assert_equal(len2, 2)






