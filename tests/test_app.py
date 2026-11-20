"""本地识别台：波形编解码、逐帧路径折段、一次端到端推理。"""

import unittest

import numpy as np

from echolens.app import base64_to_wave, emission_spans, recognize, wave_to_base64
from echolens.model import CTCNet
from echolens.synth import synth_sentence


class AppTest(unittest.TestCase):
    def test_pcm_base64_roundtrip(self):
        wave = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        back = base64_to_wave(wave_to_base64(wave))
        self.assertEqual(back.shape, (4,))
        self.assertLess(float(np.max(np.abs(back - wave))), 1e-4)

    def test_emission_spans_collapse_and_split(self):
        path = [0, 1, 1, 1, 0, 2, 0, 2, 2, 0]
        spans = emission_spans(path)
        self.assertEqual(spans[0], (1, 1, 3))
        self.assertEqual(spans[1], (2, 5, 5))
        self.assertEqual(spans[2], (2, 7, 8))  # 空白后再出现同类，另起一段

    def test_recognize_returns_three_decoders(self):
        model = CTCNet()
        wave = synth_sentence(["kato"], sr=8000, seed=1, speaker="s1")
        result = recognize(model, wave, sr=8000, ref=list("kato"), beam=4)
        self.assertEqual(
            [run["decoder"] for run in result["runs"]], ["greedy", "beam", "beam+lm"]
        )
        self.assertEqual(len(result["posteriors"]["tokens"]), 11)
        self.assertGreater(result["posteriors"]["frames_total"], 10)
        self.assertTrue(result["wave"])
        self.assertTrue(all(0.0 <= span["conf"] <= 1.0 for span in result["spans"]))


if __name__ == "__main__":
    unittest.main()
