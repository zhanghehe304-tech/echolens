"""echolens.dsp —— 共振峰滤波器与声源，只依赖 numpy。

共振峰用两极点滤波器的闭式冲激响应 + 卷积实现，不引入 scipy，也不手写递归循环。
"""

from __future__ import annotations

import numpy as np

from .tables import BANDWIDTHS, PHONES

__all__ = ["resonator_ir", "cascade", "glottal_source", "noise_source", "render_phone"]


def resonator_ir(freq: float, bandwidth: float, sr: int, max_len: int = 700) -> np.ndarray:
    """两极点共振峰的冲激响应：(1 - 2r cosω z^-1 + r^2 z^-2)^-1。

    闭式解 ``r^n sin((n+1)ω) / sinω``，当 FIR 用。
    """
    freq = float(min(freq, sr / 2.0 - 200.0))  # 别贴着奈奎斯特
    freq = float(max(freq, 80.0))
    radius = float(np.exp(-np.pi * bandwidth / sr))
    omega = 2.0 * np.pi * freq / sr
    n = np.arange(max_len, dtype=np.float64)
    h = (radius**n) * np.sin(omega * (n + 1.0)) / max(np.sin(omega), 1e-6)
    # 按包络截断到 -60 dB。不能按"首个小于阈值的采样"截：零交叉会把滤波器切成 1 ms
    tail = int(np.ceil(np.log(1e-3) / np.log(max(radius, 1e-6))) + 2)
    h = h[: max(8, min(tail, max_len))]
    return h.astype(np.float32)


def cascade(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """级联一层共振峰：卷完截回原长度，保持音素时长不变（对齐才好算）。"""
    if x.size == 0:
        return x
    full = np.convolve(x.astype(np.float64), ir.astype(np.float64))
    return full[: x.size].astype(np.float32)


def glottal_source(length: int, sr: int, f0: float, tilt: float, rng) -> np.ndarray:
    """声门脉冲串：每个基音周期一个升余弦脉冲，tilt 越大脉冲越宽（频谱越平）。"""
    source = np.zeros(length, dtype=np.float32)
    period = max(int(round(sr / max(f0, 40.0))), 4)
    width = int(max(round(period * (0.25 + 0.45 * float(np.clip(tilt, 0.0, 1.0)))), 2))
    width = min(width, period)
    pulse = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(width) / width)).astype(np.float32)
    for start in range(0, length, period):
        stop = min(start + width, length)
        if stop <= start:
            break
        source[start:stop] += pulse[: stop - start]
    if length > 4:  # 一点点周期抖动，别让脉冲串在频谱上过于干净
        source *= (1.0 + 0.15 * rng.random(length)).astype(np.float32)
    return source


def noise_source(length: int, rng) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=length).astype(np.float32)


def render_phone(phone: str, length: int, sr: int, f0: float, vtl: float, tilt: float, rng) -> np.ndarray:
    """渲染单个音素：声源 -> 共振峰级联 -> 辐射差分，长度恰为 ``length``。"""
    spec = PHONES[phone]
    kind = str(spec["kind"])
    # vtl 是声道长度倍数：声道越长，共振峰越低，所以这里用 1/vtl 缩放
    formants = [float(f) / max(vtl, 1e-3) for f in spec["f"]]
    gain = float(spec["gain"])

    if kind == "fricative":
        source = noise_source(length, rng)
        for i, freq in enumerate(formants):
            source = cascade(source, resonator_ir(freq, 700.0 + 100.0 * i, sr))
    elif kind == "plosive":
        # 前半段闭合（近似静音），后半段衰减的噪声爆破
        closure = int(length * 0.45)
        burst_len = max(length - closure, 1)
        source = np.zeros(length, dtype=np.float32)
        burst = noise_source(burst_len, rng)
        for freq in formants:
            burst = cascade(burst, resonator_ir(freq, 600.0, sr))
        decay = np.exp(-np.arange(burst_len, dtype=np.float32) / max(burst_len / 2.5, 1.0))
        source[closure:] = burst[: burst_len] * decay
    else:
        source = glottal_source(length, sr, f0, tilt, rng)
        if kind == "nasal":
            source = cascade(source, resonator_ir(formants[0], 100.0, sr))  # 鼻音的低位极强
        for i, freq in enumerate(formants):
            source = cascade(source, resonator_ir(freq, BANDWIDTHS[min(i, len(BANDWIDTHS) - 1)], sr))

    # 辐射效应：再过一次一阶差分，把高频抬起来（真实语音的 lips radiation）
    radiated = np.diff(source, prepend=source[:1]).astype(np.float32)
    peak = float(np.abs(radiated).max())
    if peak > 1e-6:
        radiated = radiated / peak * gain
    return radiated
