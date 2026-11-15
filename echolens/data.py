"""echolens.data —— 名字兼容层：合成器、语料装配与 manifest 都从这里导出。

v0.4.0 把实现拆去了 :mod:`echolens.tables` / :mod:`echolens.dsp` /
:mod:`echolens.synth` / :mod:`echolens.corpus`，这里只是为了让
``from echolens.data import PHONES, synth_dataset`` 这类老写法继续有效。
"""

from __future__ import annotations

from .corpus import read_manifest, synth_dataset, write_manifest
from .synth import synth_sentence, synth_word
from .tables import F0_BASE, PHONES, SPEAKERS, VOICES, WORDS, voice

__all__ = [
    "PHONES",
    "WORDS",
    "SPEAKERS",
    "VOICES",
    "F0_BASE",
    "voice",
    "synth_word",
    "synth_sentence",
    "synth_dataset",
    "read_manifest",
    "write_manifest",
]
