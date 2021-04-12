#!/usr/bin/env python

"""Tests for `hfcnn` package."""

import unittest
import hfcnn


class TestPackage(unittest.TestCase):
    """Tests for `hfcnn` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_version_type(self):
        """Assure that version type is str."""

        self.assertIsInstance(hfcnn.__version__, str)
