"""echolens.data —— 可控的合成音素序列数据，让整条流水线不依赖外部下载。

这不是玩具式的随机噪声。每个"音素"是一个带二次谐波的稳态音，音素之间留静音间隔，
再按说话人级别改变语速、音高和音量——和真实语音相比它太干净了，但足够让一个 CTC
模型学出真正的对齐关系，也就够用来验证流水线的每一环。

换成真数据时不用改代码：准备一个 jsonl，每行 ``{"audio": "a.wav", "text": "k a t o"}``
就是 ``echolens train --manifest`` 的输入。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .audio import write_wav

__all__ = ["PHONES", "WORDS", "synth_word", "synth_dataset", "read_manifest", "write_manifest"]

# 音素 -> 基频。拉开到 280~1700 Hz，log-mel 上的区分度足够，同时不超过 8 kHz 的奈奎斯特
PHONES: Dict[str, float] = {
    "a": 280.0,
    "i": 360.0,
    "u": 450.0,
    "e": 560.0,
    "o": 700.0,
    "k": 860.0,
    "t": 1050.0,
    "s": 1250.0,
    "m": 1450.0,
    "r": 1700.0,
}

# 词表：每个词 3~5 个音素，尽量覆盖不同起止音素，避免模型靠首音素走捷径
WORDS: Tuple[str, ...] = (
    "kato",
    "saku",
    "tima",
    "suto",
    "kemi",
    "tara",
    "soki",
    "muka",
    "tesu",
    "kora",
    "sami",
    "toku",
    "mesa",
    "kisu",
    "tamo",
    "seta",
)


def synth_word(
    word: str,
    sr: int = 8000,
    seed: int = 0,
    phone_ms: float = 120.0,
    gap_ms: float = 55.0,
    noise: float = 0.04,
) -> np.ndarray:
    """把一个词渲染成波形：音素按序拼接，每个音素带随机音高抖动与音量起伏。"""
    rng = np.random.default_rng(seed)
    rate = float(rng.uniform(0.9, 1.15))  # 语速
    amp = float(rng.uniform(0.6, 1.0))  # 音量

    pieces: List[np.ndarray] = [np.zeros(int(sr * 0.15), dtype=np.float32)]  # 前置静音
    gap = np.zeros(int(sr * gap_ms / 1000.0 * rate), dtype=np.float32)

    for phone in word:
        if phone not in PHONES:
            raise ValueError(f"未定义的音素: {phone}")
        freq = PHONES[phone] * float(rng.uniform(0.98, 1.02))
        length = int(sr * phone_ms / 1000.0 * rate * float(rng.uniform(0.9, 1.1)))
        t = np.arange(length, dtype=np.float32) / sr
        tone = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t)
        ramp = max(int(sr * 0.01), 1)
        envelope = np.ones(length, dtype=np.float32)
        envelope[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        envelope[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        pieces.append((tone * envelope * amp).astype(np.float32))
        pieces.append(gap)

    wave = np.concatenate(pieces + [np.zeros(int(sr * 0.1), dtype=np.float32)])
    wave += rng.normal(0.0, noise, size=wave.size).astype(np.float32)
    return np.clip(wave, -1.0, 1.0).astype(np.float32)


def synth_dataset(
    out_dir: str | Path,
    n_train: int = 120,
    n_dev: int = 30,
    seed: int = 0,
    sr: int = 8000,
    keep_audio: bool = True,
) -> Tuple[Path, Path]:
    """生成训练/开发两份 jsonl，返回 (train_manifest, dev_manifest)。

    音频默认落盘到 ``<out_dir>/audio``，方便直接用耳朵抽查合成结果。
    """
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    train = _write_split(audio_dir, n_train, seed, sr, keep_audio, prefix="train")
    dev = _write_split(audio_dir, n_dev, seed + 1000, sr, keep_audio, prefix="dev")

    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"
    write_manifest(train_path, train)
    write_manifest(dev_path, dev)
    return train_path, dev_path


def _write_split(audio_dir: Path, count: int, seed: int, sr: int, keep_audio: bool, prefix: str) -> List[dict]:
    rng = np.random.default_rng(seed)
    items: List[dict] = []
    for i in range(count):
        word = str(WORDS[int(rng.integers(0, len(WORDS)))])
        wave = synth_word(word, sr=sr, seed=seed + i)
        name = f"{prefix}-{i:04d}-{word}.wav"
        if keep_audio:
            write_wav(audio_dir / name, wave, sr=sr)
            # 用正斜杠，manifest 在 Windows 和 Linux 上都能被同一份代码读
            relative = f"audio/{name}"
            items.append({"audio": relative, "text": " ".join(word)})
        else:
            items.append({"audio": name, "text": " ".join(word)})
    return items


def write_manifest(path: str | Path, items: List[dict]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(path)


def read_manifest(path: str | Path) -> List[dict]:
    path = Path(path)
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "audio" not in row or "text" not in row:
                raise ValueError(f"manifest 每行需要 audio/text: {path}")
            items.append(row)
    if not items:
        raise ValueError(f"manifest 为空: {path}")
    return items
