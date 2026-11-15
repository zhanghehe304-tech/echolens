"""echolens.corpus —— 语料装配：说话人轮转切分、held-out 集合、manifest 读写。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .audio import write_wav
from .synth import synth_sentence
from .tables import SPEAKERS, WORDS

__all__ = ["synth_dataset", "write_manifest", "read_manifest"]


def synth_dataset(
    out_dir: str | Path,
    n_train: int = 120,
    n_dev: int = 30,
    seed: int = 0,
    sr: int = 8000,
    keep_audio: bool = True,
    n_speakers: int = 4,
    min_words: int = 1,
    max_words: int = 2,
    heldout_speakers: Sequence[str] | None = None,
    n_dev_unseen: int = 0,
    f0_jitter: float = 0.0,
) -> Tuple[Path, ...]:
    """生成训练/开发两份 jsonl，返回 (train_manifest, dev_manifest)。

    音频默认落盘到 ``<out_dir>/audio``，方便直接用耳朵抽查合成结果。说话人按句轮转，
    保证训练集和开发集覆盖到同一批音色。

    给了 ``heldout_speakers`` 且 ``n_dev_unseen > 0`` 时，额外生成一份
    ``dev-unseen.jsonl``（只有没进过训练的音色），此时返回三元组：
    (train, dev, dev_unseen)。这是测"跨说话人泛化"的唯一诚实的做法。
    """
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    keep = max(1, min(n_speakers, len(SPEAKERS)))
    speakers = list(SPEAKERS[:keep])
    # 音高增强只加在训练集：开发集保持干净，才量得准泛化
    train = _write_split(
        audio_dir, n_train, seed, sr, keep_audio, "train", speakers, min_words, max_words, f0_jitter
    )
    dev = _write_split(
        audio_dir, n_dev, seed + 1000, sr, keep_audio, "dev", speakers, min_words, max_words
    )

    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"
    write_manifest(train_path, train)
    write_manifest(dev_path, dev)

    if heldout_speakers and n_dev_unseen > 0:
        unseen = _write_split(
            audio_dir, n_dev_unseen, seed + 2000, sr, keep_audio, "dev-unseen",
            list(heldout_speakers), min_words, max_words,
        )
        unseen_path = out_dir / "dev-unseen.jsonl"
        write_manifest(unseen_path, unseen)
        return train_path, dev_path, unseen_path

    return train_path, dev_path


def _write_split(
    audio_dir: Path,
    count: int,
    seed: int,
    sr: int,
    keep_audio: bool,
    prefix: str,
    speakers: Sequence[str],
    min_words: int,
    max_words: int,
    f0_jitter: float = 0.0,
) -> List[dict]:
    rng = np.random.default_rng(seed)
    pool = list(speakers)
    items: List[dict] = []
    for i in range(count):
        speaker = pool[i % len(pool)]
        n_words = int(rng.integers(min_words, max_words + 1))
        words = [str(WORDS[int(rng.integers(0, len(WORDS)))]) for _ in range(n_words)]
        wave = synth_sentence(words, sr=sr, seed=seed + i, speaker=speaker, f0_jitter=f0_jitter)
        name = f"{prefix}-{i:04d}.wav"
        if keep_audio:
            write_wav(audio_dir / name, wave, sr=sr)
            relative = f"audio/{name}"  # 正斜杠：Windows 和 Linux 读同一份代码
        else:
            relative = name
        items.append(
            {"audio": relative, "text": " ".join("".join(words)), "words": words, "speaker": speaker}
        )
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
