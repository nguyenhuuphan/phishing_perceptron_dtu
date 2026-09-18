import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perceptron import Perceptron  # noqa: E402
from train_eval import SELECTION_METRIC, select_checkpoint  # noqa: E402


class ModelSelectionTests(unittest.TestCase):
    def test_epoch_callback_runs_and_fit_resets_history(self):
        x = np.array([[-1.0], [1.0]])
        y = np.array([0, 1])
        observed = []
        model = Perceptron(max_epochs=4, random_state=3)

        model.fit(
            x,
            y,
            epoch_callback=lambda epoch, updates, current: observed.append(
                (epoch, updates, current.W.copy())
            ),
        )
        self.assertEqual([item[0] for item in observed], list(range(1, len(observed) + 1)))
        self.assertEqual(len(observed), len(model.history))

        first_run_length = len(model.history)
        model.fit(x, y)
        self.assertEqual(len(model.history), first_run_length)

    def test_checkpoint_uses_validation_metric(self):
        x_train = np.array([[-1.0], [1.0], [-1.0], [1.0]])
        y_train = np.array([0, 1, 0, 1])
        x_validation = np.array([[-1.0], [1.0]])
        y_validation = np.array([0, 1])
        model = Perceptron(max_epochs=5, random_state=11)

        selection = select_checkpoint(
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
        history = selection["validation_history"]
        expected = min(
            history,
            key=lambda row: (row[SELECTION_METRIC], row["epoch"]),
        )
        self.assertEqual(selection["best_epoch"], expected["epoch"])
        self.assertEqual(
            selection["best_validation_evaluation"]["metrics"][SELECTION_METRIC],
            expected[SELECTION_METRIC],
        )

    def test_saved_report_describes_validation_selection_and_final_refit(self):
        report = json.loads(
            (ROOT / "output" / "eval_results.json").read_text(encoding="utf-8")
        )
        training = report["training"]
        history = training["validation_history"]
        expected = min(
            history,
            key=lambda row: (row[SELECTION_METRIC], row["epoch"]),
        )

        self.assertEqual(training["selection_metric"], SELECTION_METRIC)
        self.assertEqual(training["best_epoch"], expected["epoch"])
        self.assertEqual(training["best_epoch"], 9)
        self.assertEqual(training["final_fit_sources"], ["train", "validation"])
        self.assertEqual(training["final_fit_rows"], 9397)
        self.assertEqual(training["final_fit_epochs"], training["best_epoch"])


if __name__ == "__main__":
    unittest.main()
