import unittest

import torch
import torch.nn.functional as functional

from orgu.train_encoder import batch_loss, compute_pos_weight


class LossFunctionTests(unittest.TestCase):
    def test_negative_target_weight_changes_only_zero_target_weight(self):
        logits = torch.tensor([[1.0, 1.0]])
        targets = torch.tensor([[1.0, 0.0]])
        elementwise = functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        expected = (elementwise[0, 0] + 0.1 * elementwise[0, 1]) / 1.1

        observed = batch_loss(logits, targets, "L1")

        self.assertTrue(torch.allclose(observed[0], expected))
        self.assertLess(float(observed[0]), float(batch_loss(logits, targets, "L0")[0]))

    def test_positive_weights_use_training_label_frequencies(self):
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        observed = compute_pos_weight(targets)

        self.assertTrue(torch.allclose(observed, torch.tensor([2.0, 0.5])))


if __name__ == "__main__":
    unittest.main()