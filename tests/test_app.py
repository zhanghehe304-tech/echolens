"""本地识别台：波形编解码、逐帧路径折段、一次端到端推理、转写接口。"""

import base64
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np

from echolens.app import base64_to_wave, emission_spans, recognize, transcribe_audio, wave_to_base64
from echolens.model import CTCNet
from echolens.synth import synth_sentence


class _AsrStub(BaseHTTPRequestHandler):
    body = b"{}"

    def do_POST(self):  # noqa: N802 - 基类的命名
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        payload = self.__class__.body
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


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

    def test_transcribe_audio_api_engine(self):
        _AsrStub.body = json.dumps(
            {
                "text": "你好，世界",
                "language": "zh",
                "duration": 1.5,
                "segments": [{"start": 0.0, "end": 1.5, "text": "你好，世界"}],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        server = ThreadingHTTPServer(("127.0.0.1", 0), _AsrStub)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            runtime = SimpleNamespace(asr_lock=threading.Lock(), asr_cache={})
            body = {
                "engine": "api",
                "model": "whisper-1",
                "api_key": "k",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "filename": "a.wav",
                "audio": base64.b64encode(b"RIFFfake").decode("ascii"),
            }
            result = transcribe_audio(runtime, body)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result["text"], "你好，世界")
        self.assertEqual(result["engine"], "api:whisper-1")
        self.assertEqual(len(result["segments"]), 1)
        self.assertGreaterEqual(result["elapsed_ms"], 0.0)

    def test_transcribe_audio_rejects_empty(self):
        runtime = SimpleNamespace(asr_lock=threading.Lock(), asr_cache={})
        with self.assertRaises(ValueError):
            transcribe_audio(runtime, {"engine": "api", "audio": ""})

    def test_transcribe_audio_unknown_engine(self):
        runtime = SimpleNamespace(asr_lock=threading.Lock(), asr_cache={})
        body = {
            "engine": "nope",
            "audio": base64.b64encode(b"RIFF").decode("ascii"),
            "filename": "a.wav",
        }
        with self.assertRaises(Exception):
            transcribe_audio(runtime, body)


if __name__ == "__main__":
    unittest.main()
