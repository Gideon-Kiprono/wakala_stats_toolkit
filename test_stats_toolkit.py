"""Unit tests for the toolkit. Run with:  python -m unittest -v   (or pytest)."""
import unittest
import tempfile
from pathlib import Path
from stats_toolkit import load_column  # add to your import line

from stats_toolkit import mean, median, mode, variance, std_dev, quartiles, iqr_outliers, StatisticsAnalyzer


class TestCentralTendency(unittest.TestCase):
    def test_mean_of_known_values(self):
        self.assertAlmostEqual(mean([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(mean([10]), 10.0)

    def test_median_even_and_odd_lengths(self):
        self.assertEqual(median([3, 1, 2]), 2)         # odd, unsorted input
        self.assertEqual(median([4, 1, 3, 2]), 2.5)    # even: this catches your bug

    def test_mode_with_ties(self):
        self.assertEqual(mode([1, 1, 2, 2, 3]), [1, 2])   # tie
        self.assertEqual(mode([1, 2, 2, 3]), [2])         # single mode, still a list
        self.assertEqual(mode([1, 2, 3]), [])             # no mode

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            mean([])

        with self.assertRaises(ValueError):
            median([])


class TestSpread(unittest.TestCase):
    def test_sample_vs_population_variance(self):
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        self.assertAlmostEqual(variance(data, sample=False), 4.0)       # population
        self.assertAlmostEqual(variance(data, sample=True), 32 / 7)     # sample
        self.assertAlmostEqual(std_dev(data, sample=False), 2.0)

    def test_std_dev_single_value(self):
        self.assertEqual(std_dev([5], sample=False), 0.0)   # population: fine
        with self.assertRaises(ValueError):
            std_dev([5])                                     # sample: undefined


class TestOutliers(unittest.TestCase):
    def test_no_outliers_in_uniform_data(self):
        result = iqr_outliers([10, 11, 12, 13, 14, 15])
        self.assertEqual(result["outliers"], [])

    def test_detects_obvious_outlier(self):
        result = iqr_outliers([10, 11, 12, 13, 14, 100])
        self.assertEqual(result["outliers"], [100])

    def test_quartiles_known_values(self):
        self.assertEqual(quartiles([1, 2, 3, 4]), (1.75, 2.5, 3.25))
        self.assertEqual(quartiles([5]), (5, 5, 5))
        self.assertEqual(quartiles([7, 1, 3])[1], median([7, 1, 3]))  # Q2 == median


class TestLoading(unittest.TestCase):
    def test_invalid_and_missing_values_are_rejected(self):
        csv_text = (
            "agent_id,amount\n"
            "A1,100.50\n"     # row 2: valid
            "A2,  200  \n"    # row 3: valid, whitespace stripped
            "A3,abc\n"        # row 4: rejected
            "A4,\n"           # row 5: rejected (empty)
            "A5,nan\n"        # row 6: rejected
            "A6,inf\n"        # row 7: rejected
            'A7, "12,40O"\n'   # row 8: rejected ( quoted like my real file)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(csv_text, encoding="utf-8")
            valid, rejected = load_column(path, "amount")

            self.assertEqual(valid, [100.5, 200.0])
            self.assertEqual([row for row, _ in rejected], [4, 5, 6, 7, 8])
            with self.assertRaises(KeyError):
                load_column(path, "missing_column")

class TestAnalyzer(unittest.TestCase):
    def test_without_outliers_removes_extremes(self):
        a = StatisticsAnalyzer([10, 11, 12, 13, 14, 100], name="x")
        self.assertEqual(a.without_outliers().values, [10, 11, 12, 13, 14])
        self.assertEqual(a.values[-1], 100)          # original unchanged

    def test_labels_stay_aligned(self):
        labels = [("d1", "A"), ("d2", "B"), ("d3", "C"), ("d4", "D"), ("d5", "E"), ("d6", "F")]
        a = StatisticsAnalyzer([10, 11, 12, 13, 14, 100], labels=labels)
        self.assertEqual(a.outlier_details(), [(100, "d6", "F")])

    


if __name__ == "__main__":
    unittest.main()
