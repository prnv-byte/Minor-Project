"""Phase 3 preprocessing tests.

These tests run without EZKL and never use raw data over HTTP.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from infer_and_prove import preprocess_record

HERE = Path(__file__).parent


class PreprocessingTests(unittest.TestCase):
    def load(self, name):
        with (HERE / "sample_records" / name).open() as f:
            return json.load(f)

    def test_all_sample_records_have_exactly_one_hot_category_per_block(self):
        preprocessing_path = HERE.parent / "model" / "artifacts" / "preprocessing.json"
        with preprocessing_path.open() as f:
            preprocessing = json.load(f)

        for name in ["low_risk.json", "high_risk.json", "mixed_profile.json"]:
            vector = preprocess_record(self.load(name))
            self.assertEqual(vector.shape, (61,))
            self.assertTrue(np.isfinite(vector).all())

            offset = len(preprocessing["numeric_cols"])
            for col in preprocessing["categorical_cols"]:
                width = len(preprocessing["categorical_categories"][col])
                block = vector[offset:offset + width]
                self.assertEqual(float(block.sum()), 1.0, col)
                self.assertEqual(int(np.count_nonzero(block)), 1, col)
                offset += width

    def test_unknown_category_is_rejected(self):
        record = self.load("low_risk.json")
        record["housing"] = "unknown housing type"
        with self.assertRaisesRegex(ValueError, "Unknown value for 'housing'"):
            preprocess_record(record)

    def test_missing_field_is_rejected(self):
        record = self.load("low_risk.json")
        del record["age"]
        with self.assertRaisesRegex(ValueError, "Missing required fields"):
            preprocess_record(record)

    def test_non_finite_numeric_value_is_rejected(self):
        record = self.load("low_risk.json")
        record["amount"] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            preprocess_record(record)


if __name__ == "__main__":
    unittest.main()
