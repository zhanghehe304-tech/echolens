"""echolens.synth —— 把词/句拼成波形：音高、语速、语调起伏都在这里。"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .dsp import render_phone
from .tables import F0_BASE, PHONES, voice

__all__ = ["synth_word", "synth_sentence"]


def synth_word(
    word: str,
    sr: int = 8000,
    seed: int = 0,
    phone_ms: float = 120.0,
    gap_ms: float = 55.0,
    noise: Optional[float] = None,
    speaker: Optional[str] = None,
    f0_jitter: float = 0.0,
) -> np.ndarray:
    """把一个词渲染成波形：音素按序拼接，每个音素带轻微的音高与时长起伏。

    ``speaker`` 给定后，基频、声道长度、声门倾斜、语速基线、底噪都跟着这个说话人走；
    同一个说话人 + 同一个 seed 出来的波形逐采样一致。

    ``f0_jitter`` 在说话人音色之上再整体平移基频（±f0_jitter）。音素身份由共振峰决定，
    所以抖基频不会改变"说了什么"，只改变"谁在说"。
    """
    params = voice(speaker)
    rng = np.random.default_rng(seed)
    rate = float(rng.uniform(0.9, 1.15)) * params["rate"]  # 语速
    amp = float(rng.uniform(0.6, 1.0))  # 音量
    level = params["noise"] if noise is None else float(noise)
    vtl = params["vtl"]
    tilt = params["tilt"]
    f0 = F0_BASE * params["f0"] * (1.0 + float(rng.uniform(-f0_jitter, f0_jitter)))

    pieces: List[np.ndarray] = [np.zeros(int(sr * 0.15), dtype=np.float32)]  # 前置静音
    gap = np.zeros(max(int(sr * gap_ms / 1000.0 * rate), 1), dtype=np.float32)

    for phone in word:
        if phone not in PHONES:
            raise ValueError(f"未定义的音素: {phone}")
        length = max(int(sr * phone_ms / 1000.0 * rate * float(rng.uniform(0.9, 1.1))), 8)
        # 语调：每个音素上基频有 ±6% 的漂移，整句听起来不是一个死板的音高
        local_f0 = f0 * float(rng.uniform(0.94, 1.06))
        tone = render_phone(phone, length, sr, local_f0, vtl, tilt, rng)
        ramp = min(max(int(sr * 0.01), 1), tone.size)
        envelope = np.ones(tone.size, dtype=np.float32)
        envelope[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        envelope[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        pieces.append((tone * envelope * amp).astype(np.float32))
        pieces.append(gap)

    wave = np.concatenate(pieces + [np.zeros(int(sr * 0.1), dtype=np.float32)])
    wave += rng.normal(0.0, level, size=wave.size).astype(np.float32)
    return np.clip(wave, -1.0, 1.0).astype(np.float32)


def synth_sentence(
    words: Sequence[str],
    sr: int = 8000,
    seed: int = 0,
    speaker: Optional[str] = None,
    word_gap_ms: float = 140.0,
    f0_jitter: float = 0.0,
) -> np.ndarray:
    """把多个词连成一句：词内保持 55 ms 间隔，词与词之间换成长停顿 140 ms。"""
    pieces: List[np.ndarray] = []
    bridge = np.zeros(max(int(sr * word_gap_ms / 1000.0), 1), dtype=np.float32)
    for i, word in enumerate(words):
        pieces.append(synth_word(word, sr=sr, seed=seed + i * 17, speaker=speaker, f0_jitter=f0_jitter))
        if i < len(words) - 1:
            pieces.append(bridge)
    return np.concatenate(pieces).astype(np.float32)
