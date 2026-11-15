"""echolens.data —— 语料装配与 manifest：把合成器产出的句子写成可训练的 jsonl。

真正的合成在 :mod:`echolens.synth`（源-滤波器 / 共振峰）；这里负责按说话人轮转切分
训练集与开发集、把"没见过的音色"单独切一份出来、以及读写 manifest。

换成真数据时不用改代码：准备一个 jsonl，每行
``{"audio": "a.wav", "text": "k a t o", "words": ["kato"]}``
就是 ``echolens train --manifest`` 的输入。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .audio import write_wav
from .synth import F0_BASE, PHONES, SPEAKERS, VOICES, WORDS, synth_sentence, synth_word

__all__ = [
    "PHONES",
    "WORDS",
    "SPEAKERS",
    "VOICES",
    "F0_BASE",
    "synth_word",
    "synth_sentence",
    "synth_dataset",
    "read_manifest",
    "write_manifest",
]


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

    speakers = list(SPEAKERS[: max(1, min(n_speakers, len(SPEAKERS)))])
