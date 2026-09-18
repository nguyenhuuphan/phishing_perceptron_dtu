import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_contract import (  # noqa: E402
    FEATURE_DOMAINS,
    FEATURE_NAMES,
    INTERNAL_CLASS_NAMES,
    RAW_LABEL_TO_INTERNAL,
    validate_feature_values,
)
from preprocess import encode_label  # noqa: E402


class FeatureContractTests(unittest.TestCase):
    def test_contract_contains_30_ordered_features(self):
        self.assertEqual(len(FEATURE_NAMES), 30)
        self.assertEqual(list(FEATURE_DOMAINS), FEATURE_NAMES)
        self.assertEqual(FEATURE_DOMAINS["Redirect"], {0, 1})

    def test_contract_matches_arff_header(self):
        header = (ROOT / "data" / "Phishing_Dataset_full.arff").read_text(
            encoding="utf-8-sig"
        ).split("@data", 1)[0]
        attributes = re.findall(r"(?im)^@attribute\s+(\S+)\s+\{([^}]+)\}", header)
        self.assertEqual([name for name, _ in attributes[:-1]], FEATURE_NAMES)
        for name, values in attributes[:-1]:
            actual = {int(value.strip()) for value in values.split(",")}
            self.assertEqual(actual, FEATURE_DOMAINS[name], name)

    def test_raw_labels_map_to_phishing_positive_internal_labels(self):
        self.assertEqual(RAW_LABEL_TO_INTERNAL, {-1: 1, 1: 0})
        self.assertEqual(INTERNAL_CLASS_NAMES, {0: "legitimate", 1: "phishing"})
        encoded = encode_label(pd.Series([-1, 1, -1, 1]))
        self.assertEqual(encoded.tolist(), [1, 0, 1, 0])

    def test_feature_domain_validation_rejects_out_of_domain_value(self):
        frame = pd.DataFrame(
            {name: [next(iter(values))] for name, values in FEATURE_DOMAINS.items()}
        )
        frame["Favicon"] = frame["Favicon"].astype(float)
        frame.loc[0, "Favicon"] = 0.5
        with self.assertRaisesRegex(ValueError, "Favicon"):
            validate_feature_values(frame)

    def test_processed_artifacts_use_the_contract_mapping(self):
        train = pd.read_csv(ROOT / "output" / "train_processed.csv")
        test = pd.read_csv(ROOT / "output" / "test_processed.csv")
        self.assertEqual(train["Result"].value_counts().to_dict(), {0: 3422, 1: 2735})
        self.assertEqual(test["Result"].value_counts().to_dict(), {0: 2735, 1: 2163})

    def test_saved_weights_reproduce_reported_confusion_matrix(self):
        test = pd.read_csv(ROOT / "output" / "test_processed.csv")
        x = test[FEATURE_NAMES].to_numpy(dtype=float)
        y = test["Result"].to_numpy(dtype=int)
        weights = np.load(ROOT / "output" / "weights.npy")
        xb = np.hstack([x, np.ones((len(x), 1))])
        predicted = np.argmax(xb @ weights.T, axis=1)
        matrix = [
            [int(np.sum((y == actual) & (predicted == guess))) for guess in range(2)]
            for actual in range(2)
        ]
        report = json.loads((ROOT / "output" / "eval_results.json").read_text("utf-8"))
        self.assertEqual(matrix, report["confusion_matrix"])


if __name__ == "__main__":
    unittest.main()
