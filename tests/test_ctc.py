"""涉及 torch 的部分：模型形状、CTC 损失、短训练、checkpoint。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from echolens import log_mel, synth_dataset, synth_word

try:  # pragma: no cover - 取决于环境
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "未安装 torch")
class CTCTest(unittest.TestCase):
    def test_forward_shape_matches_out_lengths(self):
        from echolens.model import CTCNet, model_out_lengths

        model = CTCNet(n_mels=40, hidden=32)
        feats = torch.randn(2, 100, 40)
        lengths = torch.tensor([100, 80])
        logits = model(feats, lengths)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[2], 11)  # 10 个音素 + blank
        expected = model_out_lengths(lengths)
        self.assertEqual(list(logits.shape[1:2]), [int(expected.max())])

    def test_ctc_loss_is_finite(self):
        from echolens.model import BLANK, CTCNet, model_out_lengths

        model = CTCNet(n_mels=40, hidden=32)
        feats = torch.randn(2, 100, 40)
        lengths = torch.tensor([100, 90])
        logits = model(feats, lengths).log_softmax(dim=-1).transpose(0, 1)
        loss = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)(
            logits,
            torch.tensor([1, 2, 3, 4, 1, 2]),
            model_out_lengths(lengths),
            torch.tensor([3, 3]),
        )
        self.assertTrue(np.isfinite(float(loss.detach())))

    def test_tiny_training_and_checkpoint(self):
        from echolens.train import evaluate, load_checkpoint, load_split, save_checkpoint, train

        with tempfile.TemporaryDirectory() as directory:
            train_manifest, dev_manifest = synth_dataset(directory, n_train=24, n_dev=8, seed=0)
            feats, texts = load_split(train_manifest)
            dev_feats, dev_texts = load_split(dev_manifest)
            model, history = train(feats, texts, dev_feats, dev_texts, epochs=3, batch_size=8, verbose=False)

            self.assertEqual(len(history), 3)
            self.assertEqual(history[-1]["epoch"], 3.0)
            # 三轮只够看到损失在动，不指望收敛
            self.assertGreater(history[0]["loss"], 0.0)
            self.assertLessEqual(history[-1]["loss"], history[0]["loss"])

            summary = evaluate(model, dev_feats, dev_texts)
            self.assertGreaterEqual(float(summary["error_rate"]), 0.0)
            self.assertLessEqual(float(summary["error_rate"]), 1.5)  # 编辑距离可能超过参考长度

            path = save_checkpoint(model, Path(directory) / "m.pt")
            reloaded = load_checkpoint(path)
            again = evaluate(reloaded, dev_feats, dev_texts)
            self.assertAlmostEqual(float(again["error_rate"]), float(summary["error_rate"]))

    def test_features_feed_model(self):
        from echolens.model import CTCNet

        model = CTCNet(n_mels=40, hidden=32)
        feats = torch.from_numpy(log_mel(synth_word("saku", seed=4))[None, ...])
        lengths = torch.tensor([feats.shape[1]])
        with torch.no_grad():
            logits = model(feats, lengths)
        self.assertEqual(logits.shape[-1], 11)
        self.assertLess(logits.shape[1], feats.shape[1])  # 卷积确实降了采样


if __name__ == "__main__":
    unittest.main()
