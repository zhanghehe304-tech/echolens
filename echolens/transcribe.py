"""真实语音转文字：本地 faster-whisper，或任何 OpenAI 兼容的转写接口。

这一条路和仓库里的 10 音素玩具模型是分开的：``echolens app`` 里那套 CTC 只认
合成语音的 10 个音素，这里换成一个真正的声学模型（本地）或一个远程接口，
才是"把一段录音变成文字"。
"""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "TranscriptionError",
    "Segment",
    "Transcription",
    "ApiTranscriber",
    "LocalTranscriber",
    "build_multipart",
    "format_srt",
    "format_timestamp",
    "join_segments",
    "make_transcriber",
    "to_simplified",
]

API_KEY_ENV = "ECHOLENS_ASR_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"


class TranscriptionError(RuntimeError):
    """转写失败：缺依赖、缺密钥、接口报错，都用这个类型抛出来。"""


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcription:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None
    engine: str = ""

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "language": self.language,
            "duration": self.duration,
            "text": self.text,
            "segments": [
                {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                for s in self.segments
            ],
        }


CJK_LANGUAGES = {"zh", "ja", "ko", "yue", "w"}


def join_segments(pieces: list["Segment"], language: str | None) -> str:
    """中文/日文直接拼，其它语言用空格拼——否则英文会拼成 ``dog.This``。"""
    if not pieces:
        return ""
    separator = "" if (language or "") in CJK_LANGUAGES else " "
    return separator.join(p.text for p in pieces).strip()


def to_simplified(text: str) -> str:
    """繁体转简体：Whisper 的中文输出常年是繁体，zhconv 是纯 Python 的小依赖。"""
    try:
        import zhconv
    except ImportError as exc:  # pragma: no cover - 依赖缺失时才走到
        raise TranscriptionError("简繁转换需要 zhconv：pip install 'echolens[asr]'") from exc
    return zhconv.convert(text, "zh-hans")


def format_timestamp(seconds: float, sep: str = ",") -> str:
    """秒 -> ``00:00:01,250``（SRT 用逗号，VTT 用点）。"""
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    hours, rest = divmod(total_ms, 3600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def format_srt(transcription: Transcription) -> str:
    """按段落拼成 SRT；没有时间轴时退化成一行文本，不假装有时间码。"""
    if not transcription.segments:
        return transcription.text.strip() + "\n"
    blocks = []
    for index, seg in enumerate(transcription.segments, start=1):
        blocks.append(
            f"{index}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text.strip()}\n"
        )
    return "\n".join(blocks)


def build_multipart(
    fields: dict[str, str], file_field: str, path: Path, boundary: str | None = None
) -> tuple[bytes, str]:
    """手搓 multipart/form-data，不引 requests 之类的依赖。

    返回 (请求体字节, boundary)。文件字段的 MIME 按后缀猜，猜不出用
    application/octet-stream——OpenAI 只认后缀，其它兼容服务多数也只看能否解码。
    """
    boundary = boundary or f"echolens-{uuid.uuid4().hex}"
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    data = path.read_bytes()
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="{path.name}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode()
        + data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


class LocalTranscriber:
    """本地 faster-whisper：CPU 也能跑，第一次用会下载模型。"""

    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 5,
        vad: bool = True,
        simplified: bool = False,
    ):
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad = vad
        self.simplified = simplified
        self._engine = None

    def _load(self):
        if self._engine is not None:
            return self._engine
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - 安装缺失时才走到
            raise TranscriptionError(
                "本地转写需要 faster-whisper：pip install -e '.[asr]'（或 pip install faster-whisper）"
            ) from exc
        self._engine = WhisperModel(self.model, device=self.device, compute_type=self.compute_type)
        return self._engine

    def transcribe(self, path: str | Path) -> Transcription:
        path = Path(path)
        if not path.exists():
            raise TranscriptionError(f"音频不存在：{path}")
        engine = self._load()
        segments, info = engine.transcribe(
            str(path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad,
        )
        pieces = [
            Segment(start=float(s.start), end=float(s.end), text=str(s.text).strip()) for s in segments
        ]
        pieces = [p for p in pieces if p.text]
        language = getattr(info, "language", None)
        result = Transcription(
            text=join_segments(pieces, language),
            segments=pieces,
            language=language,
            duration=getattr(info, "duration", None),
            engine=f"whisper:{self.model}",
        )
        return _simplify(result) if self.simplified else result


class ApiTranscriber:
    """任何 OpenAI 兼容的 ``POST /audio/transcriptions`` 接口。"""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "whisper-1",
        api_key: str | None = None,
        language: str | None = None,
        timeout: float = 180.0,
        prompt: str | None = None,
        simplified: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get(API_KEY_ENV) or os.environ.get(OPENAI_KEY_ENV)
        self.language = language
        self.timeout = timeout
        self.prompt = prompt
        self.simplified = simplified

    def transcribe(self, path: str | Path) -> Transcription:
        path = Path(path)
        if not path.exists():
            raise TranscriptionError(f"音频不存在：{path}")
        if not self.api_key:
            raise TranscriptionError(
                f"没有 API key：用 --api-key，或先设置环境变量 {API_KEY_ENV}"
            )
        fields = {"model": self.model, "response_format": "verbose_json"}
        if self.language:
            fields["language"] = self.language
        if self.prompt:
            fields["prompt"] = self.prompt
        body, boundary = build_multipart(fields, "file", path)
        request = urllib.request.Request(
            f"{self.base_url}/audio/transcriptions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise TranscriptionError(f"接口返回 {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise TranscriptionError(f"连不上接口：{exc.reason}") from exc

        pieces = [
            Segment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)), text=str(s.get("text", "")).strip())
            for s in (payload.get("segments") or [])
        ]
        pieces = [p for p in pieces if p.text]
        result = Transcription(
            text=str(payload.get("text", "")).strip(),
            segments=pieces,
            language=payload.get("language"),
            duration=payload.get("duration"),
            engine=f"api:{self.model}",
        )
        return _simplify(result) if self.simplified else result


def _simplify(result: Transcription) -> Transcription:
    """整段与逐段都转一遍；分段转是为了 SRT 里也不带繁体。"""
    result.text = to_simplified(result.text)
    result.segments = [Segment(s.start, s.end, to_simplified(s.text)) for s in result.segments]
    return result


def make_transcriber(engine: str, **kwargs):
    """``engine`` 只接受 ``whisper`` / ``api``，拼错就直接说清楚。"""
    if engine == "whisper":
        allowed = {
            "model": kwargs.pop("model", "small") or "small",
            "device": kwargs.pop("device", "cpu"),
            "compute_type": kwargs.pop("compute_type", "int8"),
            "language": kwargs.pop("language", None),
            "beam_size": kwargs.pop("beam_size", 5),
            "vad": kwargs.pop("vad", True),
            "simplified": kwargs.pop("simplified", False),
        }
        kwargs.pop("base_url", None)
        kwargs.pop("api_key", None)
        return LocalTranscriber(**allowed)
    if engine == "api":
        return ApiTranscriber(
            base_url=kwargs.pop("base_url", "https://api.openai.com/v1"),
            model=kwargs.pop("model", "whisper-1") or "whisper-1",
            api_key=kwargs.pop("api_key", None),
            language=kwargs.pop("language", None),
            prompt=kwargs.pop("prompt", None),
            simplified=kwargs.pop("simplified", False),
        )
    raise TranscriptionError(f"不认识的引擎：{engine}（只支持 whisper / api）")
