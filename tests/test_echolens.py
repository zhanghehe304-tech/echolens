"""不依赖 torch 的那部分：特征、合成数据、对齐与评测、报告。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from echolens import PHONES, align, build_report, corpus_error_rate, error_rate, greedy_decode, log_mel, read_wav, resample, synth_dataset, synth_word, write_wav


class AudioTest(unittest.TestCase):
    def test_log_mel_shape_and_cmvn(self):
        wave = synth_word("kato", seed=1)
        feats = log_mel(wave, sr=8000, n_mels=40)
        self.assertEqual(feats.shape[1], 40)
        self.assertGreater(feats.shape[0], 40)  # 一句约 0.9 秒
        self.assertAlmostEqual(float(feats.mean()), 0.0, places=5)  # 默认做了 CMVN

    def test_wav_roundtrip(self):
        wave = synth_word("suto", seed=2)[:800]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.wav"
            write_wav(path, wave, sr=8000)
            rate, loaded = read_wav(path)
        self.assertEqual(rate, 8000)
        self.assertEqual(loaded.shape, wave.shape)
        self.assertLess(float(np.abs(loaded - wave).max()), 1e-3)  # 16-bit 量化误差

    def test_resample_doubles_length(self):
        wave = synth_word("tima", seed=3)
        up = resample(wave, 8000, 16000)
        self.assertAlmostEqual(len(up) / len(wave), 2.0, places=2)


class DataTest(unittest.TestCase):
    def test_synth_is_deterministic_per_seed(self):
        first = synth_word("kato", seed=7)
        second = synth_word("kato", seed=7)
        other = synth_word("kato", seed=8)
        self.assertTrue(np.allclose(first, second))
        # 不同种子连长度都会变（语速不同），只比前 1000 个采样点
        self.assertFalse(np.allclose(first[:1000], other[:1000]))

    def test_dataset_writes_manifest_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            train, dev = synth_dataset(directory, n_train=6, n_dev=3, seed=0)
            lines = Path(train).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 6)
            for line in lines:
                self.assertTrue(line.startswith('{"audio": "audio/train-'))
            audio = Path(directory) / "audio"
            self.assertEqual(len(list(audio.glob("*.wav"))), 9)
            # 文本是音素序列，空格分隔
            import json

            row = json.loads(lines[0])
            for token in row["text"].split():
                self.assertIn(token, PHONES)


class DecodeTest(unittest.TestCase):
    def test_greedy_collapses_repeats_and_blanks(self):
        # blank=0，连续相同合并，blank 断开重复
        self.assertEqual(greedy_decode([1, 1, 0, 2, 2, 0, 1], blank=0), [1, 2, 1])
        self.assertEqual(greedy_decode([], blank=0), [])
        self.assertEqual(greedy_decode([0, 0, 0], blank=0), [])


class AlignTest(unittest.TestCase):
    def test_substitution(self):
        ops = align(["a", "b", "c"], ["a", "x", "c"])
        self.assertEqual([op for op, _, _ in ops], ["match", "sub", "match"])

    def test_insertion_and_deletion(self):
        self.assertEqual([op for op, _, _ in align(["a", "b"], ["a", "b", "c"])], ["match", "match", "ins"])
        self.assertEqual([op for op, _, _ in align(["a", "b", "c"], ["a", "b"])], ["match", "match", "del"])

    def test_error_rate_is_edit_distance(self):
        err, length, ops = error_rate(["a", "b", "c"], ["a", "c"])
        self.assertEqual((err, length), (1, 3))
        self.assertEqual(len(ops), 3)

    def test_corpus_error_rate_is_weighted_by_reference_length(self):
        summary = corpus_error_rate([(["a", "b"], ["a", "b"]), (["c", "d", "e", "f"], ["c", "x", "y", "z"])])
        self.assertAlmostEqual(summary["error_rate"], 3 / 6)
        self.assertAlmostEqual(summary["exact_match"], 0.5)
        self.assertEqual(summary["samples"], 2.0)


class ReportTest(unittest.TestCase):
    def test_report_contains_sections(self):
        summary = evaluation_stub()
        html = build_report(summary, [{"epoch": 1.0, "loss": 2.0, "dev_per": 0.5, "exact_match": 0.0, "elapsed": 3.0}])
        self.assertIn("echolens report", html)
        self.assertIn("phone err rate", html)  # 指标卡
        self.assertIn("Per-sample errors", html)
        self.assertIn("替换", html)
        self.assertIn("data:image/png;base64,", html)  # 图已内联


def evaluation_stub():
    from echolens import error_rate

    ref, hyp = ["k", "a", "t", "o"], ["k", "a", "t", "a"]
    err, _, ops = error_rate(ref, hyp)
    return {
        "error_rate": 0.25,
        "exact_match": 0.0,
        "samples": 1.0,
        "details": [{"ref": ref, "hyp": hyp, "errors": err, "ops": ops}],
    }


if __name__ == "__main__":
    unittest.main()
