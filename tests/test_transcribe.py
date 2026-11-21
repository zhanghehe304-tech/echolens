"""真实转写：时间码、multipart 拼装、API 引擎解析与报错、引擎选择。"""

import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from echolens.transcribe import (
    ApiTranscriber,
    LocalTranscriber,
    Segment,
    Transcription,
    TranscriptionError,
    build_multipart,
    format_srt,
    format_timestamp,
    join_segments,
    make_transcriber,
    to_simplified,
)


class _StubHandler(http.server.BaseHTTPRequestHandler):
    status = 200
    body = b"{}"

    def do_POST(self):  # noqa: N802 - 基类的命名
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.__class__.last_headers = dict(self.headers)
        self.send_response(self.__class__.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.__class__.body)))
        self.end_headers()
        self.wfile.write(self.__class__.body)

    def log_message(self, *args):  # 静音
        pass


class TranscribeTest(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0), "00:00:00,000")
        self.assertEqual(format_timestamp(61.25), "00:01:01,250")
        self.assertEqual(format_timestamp(3725.5), "01:02:05,500")

    def test_format_srt(self):
        result = Transcription(
            text="你好世界",
            segments=[Segment(0.0, 1.5, "你好"), Segment(1.5, 3.0, "世界")],
        )
        srt = format_srt(result)
        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\n你好", srt)
        self.assertIn("2\n00:00:01,500 --> 00:00:03,000\n世界", srt)

    def test_build_multipart(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            wav.write_bytes(b"RIFFxxxx")
            body, boundary = build_multipart({"model": "whisper-1"}, "file", wav, boundary="B")
        self.assertIn(b'Content-Disposition: form-data; name="model"', body)
        self.assertIn(b"\r\n\r\nwhisper-1\r\n", body)
        self.assertIn(b'name="file"; filename="a.wav"', body)
        self.assertIn(b"Content-Type: audio/", body)
        self.assertIn(b"RIFFxxxx", body)
        self.assertTrue(body.endswith(b"--B--\r\n"))

    def test_api_parses_verbose_json(self):
        _StubHandler.status = 200
        _StubHandler.body = json.dumps(
            {
                "text": "今天天气不错",
                "language": "zh",
                "duration": 2.4,
                "segments": [
                    {"start": 0.0, "end": 1.2, "text": "今天"},
                    {"start": 1.2, "end": 2.4, "text": "天气不错"},
                ],
            }
        ).encode()
        _StubHandler.last_headers = {}
        server = http.server.HTTPServer(("127.0.0.1", 0), _StubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                audio = Path(tmp) / "b.wav"
                audio.write_bytes(b"RIFFyyyy")
                transcriber = ApiTranscriber(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="whisper-1",
                    api_key="k",
                )
                result = transcriber.transcribe(audio)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result.text, "今天天气不错")
        self.assertEqual(result.language, "zh")
        self.assertEqual([s.text for s in result.segments], ["今天", "天气不错"])
        self.assertEqual(_StubHandler.last_headers.get("Authorization"), "Bearer k")
        self.assertEqual(result.engine, "api:whisper-1")

    def test_api_error_is_wrapped(self):
        _StubHandler.status = 401
        _StubHandler.body = b'{"error": "bad key"}'
        server = http.server.HTTPServer(("127.0.0.1", 0), _StubHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                audio = Path(tmp) / "c.wav"
                audio.write_bytes(b"RIFFzzzz")
                transcriber = ApiTranscriber(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="k"
                )
                with self.assertRaises(TranscriptionError) as ctx:
                    transcriber.transcribe(audio)
        finally:
            server.shutdown()
            server.server_close()
        self.assertIn("401", str(ctx.exception))

    def test_join_segments_by_language(self):
        pieces = [Segment(0, 1, "Hello"), Segment(1, 2, "world")]
        self.assertEqual(join_segments(pieces, "en"), "Hello world")
        zh = [Segment(0, 1, "今天"), Segment(1, 2, "天气")]
        self.assertEqual(join_segments(zh, "zh"), "今天天气")

    def test_to_simplified(self):
        try:
            import zhconv  # noqa: F401
        except ImportError:
            self.skipTest("zhconv 未安装")
        self.assertEqual(to_simplified("今天天氣不錯，三樓會議室"), "今天天气不错，三楼会议室")

    def test_api_requires_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "d.wav"
            audio.write_bytes(b"RIFF")
            with mock.patch.dict("os.environ", {"ECHOLENS_ASR_KEY": "", "OPENAI_API_KEY": ""}):
                transcriber = ApiTranscriber()
                with self.assertRaises(TranscriptionError):
                    transcriber.transcribe(audio)

    def test_make_transcriber_rejects_unknown_engine(self):
        with self.assertRaises(TranscriptionError):
            make_transcriber("whisperx")

    def test_local_engine_reports_missing_dependency(self):
        with mock.patch.dict("sys.modules", {"faster_whisper": None}):
            with self.assertRaises(TranscriptionError) as ctx:
                LocalTranscriber()._load()
        self.assertIn("faster-whisper", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
