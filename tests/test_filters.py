#!/usr/bin/env python

"""Tests for filters.py."""
import unittest
from numpy.testing._private.utils import assert_equal
from hfcnn.lib import filters
from hfcnn.lib import files
import numpy as np


class TestFilters(unittest.TestCase):
    """Tests for lib/filters."""
    def setUp(self):
        # simple test data
        self.image = image = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
        self.kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1],])

        # actual heat load data
        self.data_good = files.import_file_from_local_cache('tests/resources/good.hkl')
        self.data_bad = files.import_file_from_local_cache('tests/resources/bad.hkl')

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

        
