import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_contract import FEATURE_DOMAINS, FEATURE_NAMES  # noqa: E402
from preprocess import grouped_stratified_split  # noqa: E402


def feature_keys(frame: pd.DataFrame) -> set[tuple]:
    return set(frame[FEATURE_NAMES].itertuples(index=False, name=None))


class DataSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = {
            name: pd.read_csv(ROOT / "output" / f"{name}_processed.csv")
            for name in ("train", "validation", "test")
        }
        cls.manifest = json.loads(
            (ROOT / "output" / "split_manifest.json").read_text(encoding="utf-8")
        )

    def test_all_rows_are_preserved(self):
        self.assertEqual(sum(len(frame) for frame in self.splits.values()), 11055)

    def test_feature_vectors_do_not_cross_splits(self):
        keys = {name: feature_keys(frame) for name, frame in self.splits.items()}
        self.assertTrue(keys["train"].isdisjoint(keys["validation"]))
        self.assertTrue(keys["train"].isdisjoint(keys["test"]))
        self.assertTrue(keys["validation"].isdisjoint(keys["test"]))

    def test_split_is_close_to_requested_ratios_and_class_balance(self):
        full_phishing_rate = 4898 / 11055
        target_ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
        for name, frame in self.splits.items():
            self.assertLess(abs(len(frame) / 11055 - target_ratios[name]), 0.002)
            self.assertLess(abs(frame["Result"].mean() - full_phishing_rate), 0.002)

    def test_manifest_records_duplicate_and_overlap_audit(self):
        self.assertEqual(self.manifest["unique_feature_vectors"], 5785)
        self.assertEqual(self.manifest["exact_duplicate_rows_beyond_first"], 5206)
        self.assertEqual(self.manifest["conflicting_label_feature_vectors"], 64)
        self.assertTrue(
            all(
                overlap == 0
                for overlap in self.manifest[
                    "cross_split_feature_vector_overlap"
                ].values()
            )
        )

    def test_grouped_split_is_deterministic_and_keeps_groups_together(self):
        base = {
            name: next(iter(values))
            for name, values in FEATURE_DOMAINS.items()
        }
        rows = []
        binary_features = [
            "having_IP_Address",
            "Shortining_Service",
            "having_At_Symbol",
        ]
        for group_number in range(8):
            row = base.copy()
            for bit, feature_name in enumerate(binary_features):
                values = sorted(FEATURE_DOMAINS[feature_name])
                row[feature_name] = values[(group_number >> bit) & 1]
            row["Result"] = group_number % 2
            rows.extend([row.copy(), row.copy()])

        frame = pd.DataFrame(rows, columns=FEATURE_NAMES + ["Result"])
        first = grouped_stratified_split(frame, ratios=(0.5, 0.25, 0.25), seed=7)
        second = grouped_stratified_split(frame, ratios=(0.5, 0.25, 0.25), seed=7)

        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])

        keys = {name: feature_keys(part) for name, part in first.items()}
        self.assertTrue(keys["train"].isdisjoint(keys["validation"]))
        self.assertTrue(keys["train"].isdisjoint(keys["test"]))
        self.assertTrue(keys["validation"].isdisjoint(keys["test"]))


if __name__ == "__main__":
    unittest.main()
