import unittest

from orgu.metrics import multilabel_metrics


class MultilabelMetricsTests(unittest.TestCase):
    def test_fixed_inventory_includes_unobserved_labels(self):
        report = multilabel_metrics(
            gold_sets=[{"works_for"}],
            predicted_sets=[{"works_for"}],
            labels=["works_for", "founded_by"],
        )
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["macro_f1_fixed_inventory"], 0.5)
        self.assertEqual(report["micro_f1"], 1.0)

    def test_empty_predictions_have_zero_scores(self):
        report = multilabel_metrics(
            gold_sets=[{"works_for"}],
            predicted_sets=[set()],
            labels=["works_for"],
        )
        self.assertEqual(report["micro_f1"], 0.0)
        self.assertEqual(report["macro_f1_fixed_inventory"], 0.0)


if __name__ == "__main__":
    unittest.main()