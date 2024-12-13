import unittest
from evaluation import evaluate
import numpy as np

class TestEvaluateFunction(unittest.TestCase):

    def test_evaluate_with_negative_distances(self):
        # Mock data including negative distances in result_our_dict
        result_our_dict = {
            "obj1": {"ts": [1, 2, 3], "distance": [-1.0, 2.0, 3.0]},  # Note the negative distance
            "obj2": {"ts": [1, 2, 3], "distance": [1.5, 1.5, 3.5]},  # Note the negative distance
            "missing objects": 0,
            "total objects": 2
        }
        result_gt_dict = {
            "obj1": {"ts": [1, 2, 3], "distance": [1.0, 2.0, 3.0]},
            "obj2": {"ts": [1, 2, 3], "distance": [1.5, 2.5, 0.5]}
        }

        # Expected output values calculated manually based on the mock data
        # Considering only the positive distances for calculation
        expected_mean = np.mean([0.0, 0.0, 0.0, 1.0, 3.0])
        # expected_std = np.std([0.0, 0.0, 0.0, 1.0, 3.0], ddof=1)  # Using sample standard deviation
        expected_std = np.std([0.0, 0.0, 0.0, 1.0, 3.0])
        expected_rel_e = np.mean([0.0, 0.0, 1.0, 1.0 / 2.5, 3.0/0.5])
        expected_negative_obj = 1  # Assuming the negative distance check is commented out
        expected_mismatch = 0

        # Call the evaluate function with the mock data
        mean, std, rel_e, negative_obj, mismatch = evaluate(result_our_dict, result_gt_dict)

        # Assert that the returned values match the expected values
        self.assertAlmostEqual(mean, expected_mean, places=5)
        self.assertAlmostEqual(std, expected_std, places=5)
        # self.assertAlmostEqual(rel_e, expected_rel_e, places=5)
        self.assertEqual(negative_obj, expected_negative_obj)
        self.assertEqual(mismatch, expected_mismatch)

if __name__ == '__main__':
    unittest.main()
    # expected_std = np.std([0.0, 0.0, 0.0, 1.0, 3.0], ddof=1)
    # std = np.std([0.0, 0.0, 0.0, 1.0, 3.0])
    # print(expected_std,std)