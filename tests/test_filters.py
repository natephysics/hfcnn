#!/usr/bin/env python

"""Tests for filters.py."""
import unittest
import numpy as np
from hfcnn.lib import files, filters
from numpy.testing._private.utils import assert_equal


class TestFilters(unittest.TestCase):
    """Tests for lib/filters."""
    def setUp(self):
        # simple test data
        self.image = image = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
        self.kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1],])

        # actual heat load data
        self.data_good = files.import_file_from_local_cache('tests/resources/good.hkl')
        self.data_bad = files.import_file_from_local_cache('tests/resources/bad.hkl')
        self.df = files.import_file_from_local_cache('tests/resources/test_df.hkl')

    def test_conv2d(self):
        """Testing the convolution step."""
        assert_equal(filters.conv2d(self.image, self.kernel, 2), np.array(45))
        assert_equal(filters.conv2d(self.image, self.kernel, 1), np.array([45, 72]))

    def test_data_selection(self):
        """Assure that data selection correctly identifies the good and bad
        files."""

        # verify true data
        self.assertTrue(filters.data_selection(self.data_good))

        # verify false data
        self.assertFalse(filters.data_selection(self.data_bad))

    def test_load_and_filter(self):
        """Test the combined filter imports and correctly filters the file.

        good_only_passes should only return the row with the 'good' value
        """
        good_only_passes = self.df.apply(lambda x: filters.load_and_filter(x, './tests'), axis=1)
        assert_equal(
            # the filtered value
            self.df[good_only_passes]['times'].item(), 
            # the expected value
            self.df[self.df['times'] == 'good']['times'].item()
            )
