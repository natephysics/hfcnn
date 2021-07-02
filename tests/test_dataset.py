import unittest
import numpy as np
from numpy.testing._private.utils import assert_equal
from hfcnn.dataset import HeatLoadDataset
from hfcnn import files, filters

class TestDataSetClass_Good(unittest.TestCase):
    """Tests for data set classes."""
    def setUp(self):
        # test dataframe
        self.df = 'tests/resources/test_df.hkl'
        self.data = files.import_file_from_local_cache('tests/resources/good.hkl').clip(min=0)
        self.PC1 = np.array([8698.])
        self.index = 1

    def test_HeatLoadDataset(self):

        """Testing the HeatLoadDataset class."""
        # Check __init__ method for both file import and manual import
        dataset_file = HeatLoadDataset(self.df, 'tests')

        # Check __len__ method  
        self.assertEqual(dataset_file.__len__(), 2)

        # check __getitem__ method
        item = dataset_file.__getitem__(self.index)
        self.assertSequenceEqual(item['image'].numpy().tolist(), self.data.tolist())
        self.assertEqual(item['label'].numpy(), self.PC1)

        # check apply method
        dataset = dataset_file.apply(
            filters.return_filter(*["data_selection", dataset_file.img_dir])
            )

        self.assertEqual(dataset.__len__(), 1)

        # Check split_by_program_num method
        prog_num_list1 = ['20180829.36']
        prog_num_list2 = ['20180829.32', '20180829.36']

        len1 = dataset_file.split_by_program_num(prog_num_list1).__len__()
        len2 = dataset_file.split_by_program_num(prog_num_list2).__len__()

        self.assertEqual(len1, 1)
        self.assertEqual(len2, 2)

class TestDataSetClass_Bad(TestDataSetClass_Good):
    """Tests for data set classes."""
    def setUp(self):
        self.df = 'tests/resources/test_df.hkl'
        self.data = files.import_file_from_local_cache('tests/resources/bad.hkl').clip(min=0)
        self.index = 0
        self.PC1 = np.array([8698.])


class TestDataSetClass_norm(unittest.TestCase):
    """Tests for normíng data"""
    def test_norm(self):
        # test dataframe
        self.df = 'tests/resources/test_df.hkl'
        dataset_file = HeatLoadDataset(self.df, 'tests')
        
        x, y = dataset_file.normalize()
        self.assertEqual(x, 27289.97265625)





