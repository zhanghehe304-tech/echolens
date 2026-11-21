"""echolens.app —— 本地识别台：合成一句话（或上传一段音频），看模型逐帧怎么判断。

只用标准库起 HTTP 服务（``http.server``），前端是 ``static/`` 下的三个文件，
不联网、不装额外依赖。启动后浏览器打开 ``http://127.0.0.1:8765``：

    echolens app
    echolens app --checkpoint out/_rand2/echolens.pt --port 9000
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["serve", "recognize", "transcribe_audio", "emission_spans", "main"]

STATIC_DIR = Path(__file__).with_name("static")
MAX_SECONDS = 20.0
MAX_ASR_BYTES = 20 * 1024 * 1024
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".opus", ".webm", ".aac", ".wma"}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


def wave_to_base64(wave, sr: int = 8000) -> str:
    """float 波形 -> 16-bit PCM 的 base64，前端直接塞进 ``audio`` 的 data URI。"""
    import numpy as np

    pcm = (np.clip(np.asarray(wave, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def base64_to_wave(text: str):
    """16-bit PCM 的 base64 -> float 波形（范围约 [-1, 1]）。"""
    import numpy as np

    raw = base64.b64decode(str(text))
    if len(raw) < 2:
        return np.zeros(0, dtype=np.float32)
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0).astype(np.float32)


def emission_spans(path: Sequence[int]) -> List[tuple]:
    """把逐帧 argmax 路径折成 (类别 id, 起始帧, 结束帧)，空白帧不算进任何一段。

    同一类别连续出现的帧算一段；中间隔了空白再出现，就是新的一段。
    """
    spans: List[List[int]] = []
    for frame, cls in enumerate(path):
        if int(cls) == 0:
            continue
        if spans and spans[-1][0] == cls and spans[-1][2] == frame - 1:
            spans[-1][2] = frame
        else:
            spans.append([int(cls), frame, frame])
    return [tuple(item) for item in spans]


def downsample_frames(matrix, limit: int = 240):
    """按块取最大值把帧数压到 ``limit`` 以内，热力图不至于一列挤成一条线。"""
    import numpy as np

    m = np.asarray(matrix, dtype=np.float32)
    if m.shape[0] <= limit:
        return m, 1
    step = int(np.ceil(m.shape[0] / float(limit)))
    cut = (m.shape[0] // step) * step
    blocks = m[:cut].reshape(-1, step, m.shape[1]).max(axis=1)
    return blocks.astype(np.float32), step


def _beam_with_lm(log_probs, tokens: Sequence[str], lm, weight: float, beam: int) -> List[int]:
    """beam 出 nbest，再用音素 bigram 重打分，取总分最高的一条。"""
    from .metrics import beam_search_nbest

    candidates = beam_search_nbest(log_probs, beam=beam, nbest=max(2, beam // 2))
    best_score: Optional[float] = None
    best_ids: List[int] = []
    for candidate, ctc_score in candidates:
        labels = [tokens[i - 1] for i in candidate if 1 <= i <= len(tokens)]
        total = float(ctc_score) + weight * lm.score(labels)
        if best_score is None or total > best_score:
            best_score = total
            best_ids = list(candidate)
    return best_ids


def recognize(
    model,
    wave,
    sr: int = 8000,
    ref: Optional[Sequence[str]] = None,
    tokens: Optional[Sequence[str]] = None,
    beam: int = 10,
    lm_weight: float = 0.3,
) -> Dict[str, Any]:
    """一句话的完整识别过程：三档解码、逐帧后验、音素分段与置信度、PER。"""
    import numpy as np
    import torch

    from .audio import log_mel
    from .metrics import PhoneLM, beam_search_decode, error_rate, greedy_decode
    from .model import TOKENS

    tokens = list(tokens or TOKENS)
    started = time.perf_counter()

    feats = log_mel(wave, sr=sr)
    if feats.shape[0] < 3:
        raise ValueError("音频太短，提不出一帧以上的特征")

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats[None, ...])
        lengths = torch.tensor([feats.shape[0]], dtype=torch.long)
        log_probs = model(x, lengths)[0].log_softmax(dim=-1).numpy()

    probs = np.exp(log_probs.astype(np.float64))
    path = log_probs.argmax(axis=-1).tolist()
    lm = PhoneLM()

    decoders: List[tuple] = [
        ("greedy", greedy_decode(path)),
        ("beam", beam_search_decode(log_probs, beam=beam)),
        ("beam+lm", _beam_with_lm(log_probs, tokens, lm, lm_weight, beam)),
    ]

    runs: List[Dict[str, Any]] = []
    for name, ids in decoders:
        phones = [tokens[i - 1] for i in ids if 1 <= i <= len(tokens)]
        entry: Dict[str, Any] = {"decoder": name, "phones": phones, "exact": None, "per": None, "errors": 0}
        if ref is not None:
            total_errors, total_ref, ops = error_rate(list(ref), phones)
            entry["exact"] = phones == list(ref)
            entry["per"] = (total_errors / float(total_ref)) if total_ref else 0.0
            entry["errors"] = total_errors
            entry["ops"] = [list(op) for op in ops]
        runs.append(entry)

    spans = []
    for cls, start, end in emission_spans(path):
        label = tokens[cls - 1] if 1 <= cls <= len(tokens) else f"#{cls}"
        spans.append(
            {
                "phone": label,
                "start": start,
                "end": end,
                "conf": round(float(probs[start : end + 1, cls].mean()), 3),
            }
        )

    frames, stride = downsample_frames(probs)
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "duration_ms": round(len(wave) / float(sr) * 1000.0, 1),
        "wave": wave_to_base64(wave, sr),
        "sr": sr,
        "ref": list(ref) if ref is not None else None,
        "runs": runs,
        "spans": spans,
        "posteriors": {
            "tokens": ["blank"] + tokens,
            "frames": np.round(frames, 3).tolist(),
            "stride": stride,
            "frames_total": int(log_probs.shape[0]),
        },
        "elapsed_ms": round(elapsed, 1),
    }


class Runtime:
    """把模型只加载一次，所有请求共用；推理串行，避免 CPU 上抢线程。"""

    def __init__(self, checkpoint: Path, sr: int = 8000):
        from .train import load_checkpoint

        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"找不到 checkpoint：{self.checkpoint}（先跑 echolens demo）")
        self.model = load_checkpoint(self.checkpoint)
        self.model.eval()
        self.sr = int(sr)
        self.lock = threading.Lock()
        self.asr_lock = threading.Lock()
        self.asr_cache: Dict[tuple, Any] = {}
        self.params = int(sum(p.numel() for p in self.model.parameters()))


def _asr_meta() -> Dict[str, Any]:
    import importlib.util

    def has(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    return {
        "whisper": has("faster_whisper"),
        "zhconv": has("zhconv"),
        "models": ["tiny", "base", "small", "medium", "large-v3"],
        "default_model": "small",
        "languages": [
            ["", "自动识别"], ["zh", "中文"], ["en", "英文"], ["ja", "日文"], ["ko", "韩文"],
        ],
        "max_bytes": MAX_ASR_BYTES,
    }


def transcribe_audio(runtime: Runtime, body: Dict[str, Any]) -> Dict[str, Any]:
    """真实转写：把前端传上来的原始音频字节落到临时文件，交给 whisper / API 引擎。"""
    from .transcribe import TranscriptionError, make_transcriber

    started = time.perf_counter()
    raw = base64.b64decode(str(body.get("audio") or ""))
    if not raw:
        raise ValueError("没有收到音频")
    if len(raw) > MAX_ASR_BYTES:
        raise ValueError(f"音频太大（上限 {MAX_ASR_BYTES // (1024 * 1024)} MB）")

    engine = str(body.get("engine") or "whisper")
    language = str(body.get("language") or "") or None
    simplified = bool(body.get("simplified"))
    suffix = Path(str(body.get("filename") or "audio.wav")).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".wav"

    fd, temp_name = tempfile.mkstemp(prefix="echolens-asr-", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        if engine == "whisper":
            model = str(body.get("model") or "small")
            device = str(body.get("device") or "cpu")
            compute_type = str(body.get("compute_type") or "int8")
            key = (model, device, compute_type)
            with runtime.asr_lock:  # faster-whisper 的模型不适合并发推理
                transcriber = runtime.asr_cache.get(key)
                if transcriber is None:
                    transcriber = make_transcriber(
                        "whisper",
                        model=model,
                        device=device,
                        compute_type=compute_type,
                        language=language,
                        simplified=simplified,
                        vad=bool(body.get("vad", True)),
                    )
                    runtime.asr_cache[key] = transcriber
                transcriber.language = language
                transcriber.simplified = simplified
                result = transcriber.transcribe(temp_name)
        elif engine == "api":
            transcriber = make_transcriber(
                "api",
                model=str(body.get("model") or "whisper-1"),
                base_url=str(body.get("base_url") or "https://api.openai.com/v1"),
                api_key=body.get("api_key") or None,
                language=language,
                prompt=body.get("prompt") or None,
                simplified=simplified,
            )
            result = transcriber.transcribe(temp_name)
        else:
            raise TranscriptionError(f"不认识的引擎：{engine}")
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass

    return {
        "engine": result.engine,
        "text": result.text,
        "language": result.language,
        "duration": result.duration,
        "segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
            for s in result.segments
        ],
        "bytes": len(raw),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


def _meta(runtime: Runtime) -> Dict[str, Any]:
    from . import __version__
    from .data import SPEAKERS, VOICES, WORDS

    return {
        "version": __version__,
        "checkpoint": runtime.checkpoint.name,
        "checkpoint_path": str(runtime.checkpoint),
        "params": runtime.params,
        "sr": runtime.sr,
        "max_seconds": MAX_SECONDS,
        "speakers": [
            {"id": name, "seen": index < 4, "f0": VOICES[name]["f0"], "vtl": VOICES[name]["vtl"]}
            for index, name in enumerate(SPEAKERS)
        ],
        "words": list(WORDS),
        "asr": _asr_meta(),
    }


def _synth_wave(words: Sequence[str], speaker: str, seed: int, snr: Optional[float], sr: int):
    from .audio import add_noise
    from .data import WORDS
    from .synth import synth_sentence

    picked = [str(w) for w in words]
    if not picked:
        raise ValueError("至少选一个词")
    for word in picked:
        if word not in WORDS:
            raise ValueError(f"词表里没有 {word}")
    wave = synth_sentence(picked, sr=sr, seed=int(seed), speaker=speaker)
    if snr is not None:
        wave = add_noise(wave, float(snr), seed=int(seed))
    ref = [ch for word in picked for ch in word]
    return wave, ref


def _static_path(raw: str) -> Optional[Path]:
    name = raw.split("?", 1)[0].lstrip("/")
    if not name.startswith("static/"):
        return None
    target = (STATIC_DIR / name[len("static/") :]).resolve()
    if STATIC_DIR.resolve() not in target.parents and target.parent != STATIC_DIR.resolve():
        return None
    if not target.is_file():
        return None
    return target


def _make_handler(runtime: Runtime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "echolens"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:  # 别把每条请求都喷到终端
            return

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html", "/index"):
                html = STATIC_DIR / "index.html"
                self._send(200, html.read_bytes(), CONTENT_TYPES[".html"])
                return
            if path == "/api/meta":
                self._json(200, {"ok": True, **_meta(runtime)})
                return
            if path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
                return
            target = _static_path(self.path)
            if target is not None:
                ctype = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
                self._send(200, target.read_bytes(), ctype)
                return
            self._json(404, {"ok": False, "error": f"没有这个路径：{path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path not in ("/api/recognize", "/api/transcribe"):
                self._json(404, {"ok": False, "error": f"没有这个接口：{path}"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    raise ValueError("空请求体")
                if length > 32 * 1024 * 1024:
                    raise ValueError("请求体太大（上限 32 MB）")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                payload = self._recognize(body) if path == "/api/recognize" else transcribe_audio(runtime, body)
                self._json(200, {"ok": True, **payload})
            except BrokenPipeError:
                return
            except Exception as exc:  # 任何错误都回成 JSON，前端好显示
                self._json(400, {"ok": False, "error": str(exc)})

        def _recognize(self, body: Dict[str, Any]) -> Dict[str, Any]:
            from .audio import resample

            mode = str(body.get("mode") or "synth")
            echo: Dict[str, Any] = {"mode": mode}
            if mode == "pcm":
                sr_in = int(body.get("sr") or runtime.sr)
                wave = base64_to_wave(str(body.get("samples") or ""))
                if wave.size == 0:
                    raise ValueError("没有收到音频采样")
                if wave.size > MAX_SECONDS * sr_in:
                    raise ValueError(f"音频太长（上限 {MAX_SECONDS:g} 秒）")
                if sr_in != runtime.sr:
                    wave = resample(wave, sr_in, runtime.sr)
                text = body.get("text")
                ref = list(str(text).replace(" ", "")) if text else None
            else:
                speaker = str(body.get("speaker") or "s1")
                seed = int(body.get("seed") or 0)
                snr = body.get("snr")
                snr = None if snr is None or snr == "" else float(snr)
                wave, ref = _synth_wave(body.get("words") or [], speaker, seed, snr, runtime.sr)
                echo.update({"speaker": speaker, "seed": seed, "snr": snr, "words": body.get("words") or []})

            with runtime.lock:
                result = recognize(
                    runtime.model,
                    wave,
                    sr=runtime.sr,
                    ref=ref,
                    beam=int(body.get("beam") or 10),
                    lm_weight=float(body.get("lm_weight") or 0.3),
                )
            result.update(echo)
            result["checkpoint"] = runtime.checkpoint.name
            return result

    return Handler


def serve(
    checkpoint: str | Path = "out/echolens.pt",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    sr: int = 8000,
) -> None:
    runtime = Runtime(Path(checkpoint), sr=sr)
    handler = _make_handler(runtime)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"echolens app: {url}")
    print(f"checkpoint: {runtime.checkpoint}（{runtime.params:,} 参数）")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    print("Ctrl+C 退出")
    try:
        server.serve_forever(poll_interval=0.4)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.shutdown()
        server.server_close()


def _parse_args(argv: Optional[Sequence[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(prog="echolens app", description="本地识别台（浏览器界面）")
    parser.add_argument("--checkpoint", default="out/echolens.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        serve(args.checkpoint, host=args.host, port=args.port, open_browser=not args.no_browser, sr=args.sr)
    except OSError as exc:
        print(f"起不来：{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
