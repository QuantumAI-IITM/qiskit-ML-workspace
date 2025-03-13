# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Modified Ad Hoc Data """

import unittest
from unittest.mock import patch
import numpy as np
from ddt import ddt, data, unpack

from test import QiskitMachineLearningTestCase

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.datasets import ad_hoc_data


@ddt
class TestModifiedAdHocData(QiskitMachineLearningTestCase):
    """Tests for the modified Ad Hoc Data implementation."""

    def setUp(self):
        """Set up test cases."""
        super().setUp()
        # Set a fixed seed for reproducible tests
        algorithm_globals.random_seed = 12345

    def test_output_shapes_default(self):
        """Test if the function returns arrays with expected shapes using default parameters."""
        training_size = 20
        test_size = 10
        n = 2
        gap = 0.3
        
        train_features, train_labels, test_features, test_labels = ad_hoc_data(
            training_size=training_size, test_size=test_size, n=n, gap=gap
        )
        
        # Check feature shapes
        self.assertEqual(train_features.shape, (training_size, n))
        self.assertEqual(test_features.shape, (test_size, n))
        
        # Check label shapes
        self.assertEqual(train_labels.shape, (training_size,))
        self.assertEqual(test_labels.shape, (test_size,))
        
        # Ensure all feature values are within the expected range (0, 2π]
        self.assertTrue(np.all(train_features > 0))
        self.assertTrue(np.all(train_features <= 2 * np.pi))
        self.assertTrue(np.all(test_features > 0))
        self.assertTrue(np.all(test_features <= 2 * np.pi))

    @data(
        (10, 5, 2, 0.1, 0),     # Sobol sampling with n=2
        (10, 5, 3, 0.2, 0),     # Sobol sampling with n=3
        (10, 5, 4, 0.3, 0),     # Sobol sampling with n=4
        (10, 5, 5, 0.1, 0),     # Sobol sampling with n=5
        (10, 5, 2, 0.2, 10),    # Latin Hypercube with n=2
        (10, 5, 3, 0.3, 10),    # Latin Hypercube with n=3
        (10, 5, 4, 0.1, 10),    # Latin Hypercube with n=4
        (10, 5, 5, 0.2, 10),    # Latin Hypercube with n=5
    )
    @unpack
    def test_dimensions(self, training_size, test_size, n, gap, divisions):
        """Test if the function correctly handles different dimensions and sampling methods."""
        train_features, train_labels, test_features, test_labels = ad_hoc_data(
            training_size=training_size, 
            test_size=test_size, 
            n=n, 
            gap=gap,
            divisions=divisions
        )
        
        # Check feature dimensions
        self.assertEqual(train_features.shape, (training_size, n))
        self.assertEqual(test_features.shape, (test_size, n))
        
        # Check if labels are correct (-1 or 1)
        self.assertTrue(np.all(np.abs(train_labels) == 1))
        self.assertTrue(np.all(np.abs(test_labels) == 1))

    @data(
        (10, 5, 2, 0.1, False),
        (10, 5, 2, 0.1, True),
    )
    @unpack
    def test_one_hot_encoding(self, training_size, test_size, n, gap, one_hot):
        """Test if one-hot encoding works correctly."""
        train_features, train_labels, test_features, test_labels = ad_hoc_data(
            training_size=training_size, 
            test_size=test_size, 
            n=n, 
            gap=gap,
            one_hot=one_hot
        )
        
        if one_hot:
            # Check if labels are one-hot encoded
            self.assertEqual(train_labels.shape, (training_size, 2))
            self.assertEqual(test_labels.shape, (test_size, 2))
            self.assertTrue(np.all(np.sum(train_labels, axis=1) == 1))
            self.assertTrue(np.all(np.sum(test_labels, axis=1) == 1))
            self.assertTrue(np.all(np.isin(train_labels, [0, 1])))
            self.assertTrue(np.all(np.isin(test_labels, [0, 1])))
        else:
            # Check if labels are -1 and 1
            self.assertEqual(train_labels.shape, (training_size,))
            self.assertEqual(test_labels.shape, (test_size,))
            self.assertTrue(np.all(np.abs(train_labels) == 1))
            self.assertTrue(np.all(np.abs(test_labels) == 1))

    def test_include_sample_total(self):
        """Test if the include_sample_total parameter works correctly."""
        training_size = 10
        test_size = 5
        n = 2
        gap = 0.1
        
        result = ad_hoc_data(
            training_size=training_size, 
            test_size=test_size, 
            n=n, 
            gap=gap,
            include_sample_total=True
        )
        
        # Check if the function returns 5 values instead of 4
        self.assertEqual(len(result), 5)
        
        train_features, train_labels, test_features, test_labels, total_samples = result
        
        # Check if total_samples is an integer and at least equal to training_size + test_size
        self.assertIsInstance(total_samples, (int, np.integer))
        self.assertGreaterEqual(total_samples, training_size + test_size)

    def test_gap_impact(self):
        """Test if the gap parameter correctly influences the separation of data."""
        n = 2
        training_size = 20
        test_size = 0  # Only testing training data
        
        # Get data with a small gap
        train_features_small_gap, train_labels_small_gap, _, _ = ad_hoc_data(
            training_size=training_size, 
            test_size=test_size, 
            n=n, 
            gap=0.1
        )
        
        # Get data with a larger gap
        train_features_large_gap, train_labels_large_gap, _, _ = ad_hoc_data(
            training_size=training_size, 
            test_size=test_size, 
            n=n, 
            gap=0.5,
            # Use the same random seed
        )
        
        # The expectation is that generating samples with a larger gap might require 
        # evaluating more potential samples before finding valid ones that satisfy 
        # the gap constraint. This is hard to test directly, but we can verify that
        # the resulting datasets contain the expected number of samples.
        self.assertEqual(len(train_features_small_gap), training_size)
        self.assertEqual(len(train_features_large_gap), training_size)

    @patch('qiskit_machine_learning.datasets.ad_hoc_data._plot_ad_hoc_data')
    def test_plot_data_flag(self, mock_plot):
        """Test if the plot_data flag calls the plotting function correctly."""
        # Test with plot_data=True and n=2
        ad_hoc_data(training_size=10, test_size=5, n=2, gap=0.1, plot_data=True)
        self.assertEqual(mock_plot.call_count, 1)
        
        # Test with plot_data=False
        mock_plot.reset_mock()
        ad_hoc_data(training_size=10, test_size=5, n=2, gap=0.1, plot_data=False)
        self.assertEqual(mock_plot.call_count, 0)
        
        # Test with plot_data=True but n>3 (should disable plotting)
        mock_plot.reset_mock()
        ad_hoc_data(training_size=10, test_size=5, n=4, gap=0.1, plot_data=True)
        self.assertEqual(mock_plot.call_count, 0)

    def test_reproducibility(self):
        """Test if setting the same random seed produces the same results."""
        algorithm_globals.random_seed = 42
        result1 = ad_hoc_data(training_size=10, test_size=5, n=2, gap=0.1)
        
        algorithm_globals.random_seed = 42
        result2 = ad_hoc_data(training_size=10, test_size=5, n=2, gap=0.1)
        
        # Check that all arrays are identical
        for arr1, arr2 in zip(result1, result2):
            np.testing.assert_array_equal(arr1, arr2)
        
        # Change seed and verify results differ
        algorithm_globals.random_seed = 43
        result3 = ad_hoc_data(training_size=10, test_size=5, n=2, gap=0.1)
        
        # At least one array should be different
        any_different = False
        for arr1, arr3 in zip(result1, result3):
            if not np.array_equal(arr1, arr3):
                any_different = True
                break
        
        self.assertTrue(any_different, "Results should differ with different random seeds")

    def test_helper_functions(self):
        """Test that helper functions work correctly."""
        # Test _n_hadamard
        from qiskit_machine_learning.datasets.ad_hoc_data import _n_hadamard
        h2 = _n_hadamard(2)
        self.assertEqual(h2.shape, (4, 4))
        
        # Test _onehot_labels
        from qiskit_machine_learning.datasets.ad_hoc_data import _onehot_labels
        labels = np.array([-1, 1, -1, 1])
        one_hot = _onehot_labels(labels)
        expected = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        np.testing.assert_array_equal(one_hot, expected)
        
        # Test _i_z
        from qiskit_machine_learning.datasets.ad_hoc_data import _i_z
        z0 = _i_z(0, 2)
        expected_z0 = np.array([1, -1, 1, -1])  # Diagonal of Z⊗I
        np.testing.assert_array_equal(np.diag(z0), expected_z0)
        
        z1 = _i_z(1, 2)
        expected_z1 = np.array([1, 1, -1, -1])  # Diagonal of I⊗Z
        np.testing.assert_array_equal(np.diag(z1), expected_z1)


if __name__ == "__main__":
    unittest.main()
