"""不依赖 torch 的那部分：特征、合成数据、对齐与评测、报告。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from echolens import (
    PHONES,
    PhoneLM,
    add_noise,
    align,
    beam_search_decode,
    beam_search_nbest,
    build_report,
    corpus_error_rate,
    error_rate,
    greedy_decode,
    log_mel,
    read_wav,
    resample,
    synth_dataset,
    synth_sentence,
    synth_word,
    word_error_rate,
    write_wav,
)


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


class FormantTest(unittest.TestCase):
    """音素身份必须来自共振峰，而不是"每个音素一个正弦"。"""

    @staticmethod
    def _envelope(wave: np.ndarray, sr: int = 8000) -> tuple[np.ndarray, np.ndarray]:
        # 取音素主体（跳过 150 ms 前导静音），做谱包络：谐波细结构不该主导结论
        segment = wave[1300:2500] * np.hanning(1500 - 300)
        spectrum = np.abs(np.fft.rfft(segment, n=4096))
        freqs = np.fft.rfftfreq(4096, d=1.0 / sr)
        smooth = np.convolve(spectrum, np.ones(31) / 31.0, mode="same")
        return freqs, smooth

    @classmethod
    def _peak(cls, wave: np.ndarray, low: float, high: float) -> float:
        freqs, smooth = cls._envelope(wave)
        band = (freqs >= low) & (freqs <= high)
        return float(freqs[band][int(np.argmax(smooth[band]))])

    def test_vowel_peaks_land_near_the_design_values(self):
        # 设计值 F1 760 / F2 1180；8 kHz 下 f0 只有百赫兹，谐波会把它拉偏一些，给 15%
        wave = synth_word("a", seed=5, phone_ms=200)
        self.assertLess(abs(self._peak(wave, 150.0, 900.0) - 760.0) / 760.0, 0.15)
        self.assertLess(abs(self._peak(wave, 900.0, 2100.0) - 1180.0) / 1180.0, 0.15)

    def test_front_and_back_vowels_are_separated_by_f2(self):
        # i 是高前元音（F2 2300 左右），u 是高后元音（F2 900 左右）：相对关系必须成立
        front = self._peak(synth_word("i", seed=6, phone_ms=200), 900.0, 3000.0)
        back = self._peak(synth_word("u", seed=6, phone_ms=200), 900.0, 3000.0)
        self.assertGreater(front, back + 400.0)

    def test_longer_vocal_tract_lowers_formants(self):
        # 声道更长（s6, vtl 1.14）-> 共振峰整体下移；声道更短（s4, vtl 0.88）-> 上移。
        # 这是"说话人音色"在本版里唯一该动的东西，测它比测任何指标都直接
        short_tract = self._peak(synth_word("a", seed=7, speaker="s4", phone_ms=200), 150.0, 900.0)
        long_tract = self._peak(synth_word("a", seed=7, speaker="s6", phone_ms=200), 150.0, 900.0)
        self.assertLess(long_tract, short_tract)


class NoiseTest(unittest.TestCase):
    def test_add_noise_hits_the_target_snr(self):
        wave = synth_word("kato", seed=1)
        for target in (20.0, 10.0, 0.0):
            noisy = add_noise(wave, target, seed=3)
            residual = noisy - wave
            measured = 10.0 * np.log10(np.mean(np.square(wave)) / np.mean(np.square(residual)))
            self.assertLess(abs(measured - target), 1.0)

    def test_add_noise_is_reproducible(self):
        wave = synth_word("suto", seed=2)
        self.assertTrue(np.allclose(add_noise(wave, 5.0, seed=4), add_noise(wave, 5.0, seed=4)))

    def test_noise_leaves_an_empty_signal_alone(self):
        empty = np.zeros(0, dtype=np.float32)
        self.assertEqual(add_noise(empty, 10.0).size, 0)


class SpeakerTest(unittest.TestCase):
    def test_same_speaker_is_deterministic(self):
        first = synth_word("kato", seed=3, speaker="s1")
        second = synth_word("kato", seed=3, speaker="s1")
        self.assertTrue(np.allclose(first, second))

    def test_different_speakers_sound_different(self):
        # 同一句话、同一个 seed，只换说话人：波形必须有可见差异（音色不是一个随机种子）
        s1 = synth_word("kato", seed=5, speaker="s1")
        s4 = synth_word("kato", seed=5, speaker="s4")
        shorter = min(len(s1), len(s4))
        self.assertFalse(np.allclose(s1[:shorter], s4[:shorter]))
        # 谱重心也拉开了：高基频说话人在高频段能量占比更高
        def centroid(wave):
            spectrum = np.abs(np.fft.rfft(wave[: len(wave) // 2]))
            freqs = np.fft.rfftfreq(len(wave) // 2, d=1.0 / 8000)
            return float((spectrum * freqs).sum() / (spectrum.sum() + 1e-9))

        self.assertNotAlmostEqual(centroid(s1), centroid(s4), places=1)

    def test_unknown_speaker_is_rejected(self):
        with self.assertRaises(ValueError):
            synth_word("kato", seed=1, speaker="s9")

    def test_sentence_holds_all_words(self):
        wave = synth_sentence(["kato", "suto"], seed=2, speaker="s2")
        single = synth_word("kato", seed=2, speaker="s2")
        self.assertGreater(len(wave), len(single))  # 两个词比一个词长
        self.assertGreater(len(wave), 2 * len(single) * 0.9)


class PitchJitterTest(unittest.TestCase):
    def test_jitter_keeps_length_and_changes_content(self):
        plain = synth_word("kato", seed=3, speaker="s1")
        jittered = synth_word("kato", seed=3, speaker="s1", f0_jitter=0.12)
        # 抖动只动基频，时长和音量序列不变
        self.assertEqual(len(plain), len(jittered))
        self.assertFalse(np.allclose(plain, jittered))

    def test_zero_jitter_is_the_plain_signal(self):
        plain = synth_word("sami", seed=7, speaker="s2")
        self.assertTrue(np.allclose(plain, synth_word("sami", seed=7, speaker="s2", f0_jitter=0.0)))


class ManifestTest(unittest.TestCase):
    def test_manifest_carries_words_and_speaker(self):
        with tempfile.TemporaryDirectory() as directory:
            train, _ = synth_dataset(directory, n_train=8, n_dev=2, seed=1, n_speakers=4)
            import json

            rows = [json.loads(line) for line in Path(train).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 8)
            speakers = {row["speaker"] for row in rows}
            self.assertEqual(len(speakers), 4)  # 8 句按说话人轮转，四个都覆盖到
            for row in rows:
                self.assertIsInstance(row["words"], list)
                self.assertGreaterEqual(len(row["words"]), 1)
                # words 拼起来必须等于 text 里的音素
                self.assertEqual("".join(row["words"]), "".join(row["text"].split()))

    def test_heldout_split_never_shares_a_speaker_with_training(self):
        with tempfile.TemporaryDirectory() as directory:
            train, dev, unseen = synth_dataset(
                directory,
                n_train=8,
                n_dev=4,
                seed=2,
                n_speakers=2,
                heldout_speakers=["s3", "s4"],
                n_dev_unseen=4,
            )
            import json

            def speakers_of(path):
                return {json.loads(line)["speaker"] for line in Path(path).read_text(encoding="utf-8").splitlines()}

            trained = speakers_of(train) | speakers_of(dev)
            self.assertEqual(trained, {"s1", "s2"})
            self.assertEqual(speakers_of(unseen), {"s3", "s4"})  # 训练从没听过这两个音色
            self.assertTrue(trained.isdisjoint(speakers_of(unseen)))


class DecodeTest(unittest.TestCase):
    def test_greedy_collapses_repeats_and_blanks(self):
        # blank=0，连续相同合并，blank 断开重复
        self.assertEqual(greedy_decode([1, 1, 0, 2, 2, 0, 1], blank=0), [1, 2, 1])
        self.assertEqual(greedy_decode([], blank=0), [])
        self.assertEqual(greedy_decode([0, 0, 0], blank=0), [])

    def test_beam_matches_greedy_on_peaked_frames(self):
        # 造一段概率很尖的帧：前两帧 1 类占优、后三帧 2 类占优，其余接近零
        frames = np.full((5, 4), -20.0)
        frames[:2, 1] = -0.01
        frames[2:, 2] = -0.01
        frames[:, 0] = -10.0
        greedy = greedy_decode(frames.argmax(axis=1).tolist(), blank=0)
        beam = beam_search_decode(frames, beam=10, blank=0)
        self.assertEqual(greedy, [1, 2])
        self.assertEqual(beam, [1, 2])

    def test_beam_keeps_single_token_for_one_phone(self):
        frames = np.full((4, 4), -20.0)
        frames[:, 1] = -0.01
        self.assertEqual(beam_search_decode(frames, beam=10, blank=0), [1])

    def test_beam_handles_empty_frames(self):
        self.assertEqual(beam_search_decode(np.zeros((0, 4)), beam=10, blank=0), [])

    def test_beam_prefers_likely_prefix_over_shorter_one(self):
        # 第 3 帧明显指向 3 类，但前两帧指向 1、2 类；beam 不该为了"更短"把 3 类丢掉
        frames = np.full((3, 4), -20.0)
        frames[0, 1] = -0.01
        frames[1, 2] = -0.01
        frames[2, 3] = -0.01
        frames[:, 0] = -12.0
        self.assertEqual(beam_search_decode(frames, beam=5, blank=0), [1, 2, 3])

    def test_nbest_is_ranked_and_starts_with_the_best(self):
        frames = np.full((4, 4), -20.0)
        frames[0, 1] = -0.01
        frames[1, 2] = -0.5
        frames[2, 1] = -0.6
        frames[3, 3] = -0.7
        frames[:, 0] = -3.0
        candidates = beam_search_nbest(frames, beam=6, blank=0, nbest=3)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0][0], beam_search_decode(frames, beam=6, blank=0))
        scores = [score for _, score in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))  # 按概率从高到低


class LanguageModelTest(unittest.TestCase):
    def test_lm_prefers_a_sequence_the_word_list_contains(self):
        lm = PhoneLM()
        in_vocab = lm.score(list("kato"))  # k-a-t-o 在词表里
        broken = lm.score(list("kaao"))  # a 后面跟 a，词表里没有这种接续
        self.assertGreater(in_vocab, broken)

    def test_lm_scores_are_finite_log_probs(self):
        lm = PhoneLM()
        self.assertLess(lm.score(list("suto")), 0.0)  # log 概率为负
        weird = lm.score(list("xyz"))
        self.assertTrue(np.isfinite(weird))


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

    def test_word_error_rate_uses_reference_word_boundaries(self):
        ref = ["k", "a", "t", "o", "s", "u", "t", "o"]
        hyp = ["k", "a", "t", "a", "s", "u", "t", "o"]  # 第一个词里替换了一个音素
        ops = align(ref, hyp)
        bad, total, flags = word_error_rate(["kato", "suto"], ops)
        self.assertEqual((bad, total), (1, 2))
        self.assertEqual(flags, [True, False])

    def test_word_error_rate_flags_word_on_deletion(self):
        ref = ["k", "a", "t", "o", "s", "u", "t", "o"]
        hyp = ["k", "a", "s", "u", "t", "o"]  # 第一个词漏了两个音素
        bad, total, _ = word_error_rate(["kato", "suto"], align(ref, hyp))
        self.assertEqual((bad, total), (1, 2))

    def test_word_error_rate_rejects_mismatched_ops(self):
        # 词长度和参考音素数对不上，说明传错了句子的对齐结果，必须报错而不是猜
        with self.assertRaises(ValueError):
            word_error_rate(["kato"], align(["k", "a", "t", "o", "s"], ["k", "a", "t", "o", "s"]))


class ReportTest(unittest.TestCase):
    def test_report_contains_sections(self):
        summary = evaluation_stub()
        html = build_report(summary, [{"epoch": 1.0, "loss": 2.0, "dev_per": 0.5, "exact_match": 0.0, "elapsed": 3.0}])
        self.assertIn("echolens report", html)
        self.assertIn("phone err rate", html)  # 指标卡
        self.assertIn("Per-sample errors", html)
        self.assertIn("替换", html)
        self.assertIn("data:image/png;base64,", html)  # 图已内联

    def test_report_shows_decoder_comparison_and_wer(self):
        summary = evaluation_stub()
        summary["word_error_rate"] = 0.5
        summary["decoder"] = "beam"
        reference = dict(summary)
        reference["decoder"] = "greedy"
        html = build_report(summary, [], reference_run=reference)
        self.assertIn("Decoder comparison", html)
        self.assertIn("word err rate", html)
        self.assertIn("greedy", html)
        self.assertIn("beam", html)

    def test_report_shows_noise_robustness_table(self):
        summary = evaluation_stub()
        summary["decoder"] = "beam"
        loud = dict(summary)
        loud["error_rate"] = 0.6
        html = build_report(summary, [], noise_runs=[("clean", summary), ("0 dB", loud)])
        self.assertIn("Noise robustness", html)
        self.assertIn("0 dB", html)

    def test_report_colors_words_the_right_way_round(self):
        # 词标记容易写反：flags 为 True 表示"这个词错了"，渲染时必须红，不是绿
        summary = evaluation_stub()
        summary["word_error_rate"] = 0.5
        summary["details"][0]["words"] = ["kato", "suto"]
        summary["details"][0]["word_flags"] = [False, True]
        html = build_report(summary, [])
        self.assertIn('<span class="ok">kato</span>', html)
        self.assertIn('<span class="sub">suto</span>', html)


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
