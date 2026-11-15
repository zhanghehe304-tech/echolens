"""echolens.audio —— WAV 读写与 log-mel 特征，只依赖 numpy。

设计原则：特征层不依赖 torch，也不假设采样率。默认参数按 8 kHz、25 ms 窗、
10 ms 帧移给出，这是语音识别里最常见的一组；别的采样率照样能跑。
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["read_wav", "write_wav", "resample", "mel_filterbank", "log_mel"]


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """读 16-bit PCM WAV，返回 (采样率, float32 单声道波形，范围约 [-1, 1])。"""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if width != 2:
        raise ValueError(f"只支持 16-bit PCM，当前 {width * 8}-bit: {path}")

    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return rate, data


def write_wav(path: str | Path, samples: np.ndarray, sr: int = 8000) -> str:
    """把 [-1, 1] 的波形写成 16-bit 单声道 WAV，方便用耳朵复核合成数据。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sr))
        handle.writeframes(pcm.tobytes())
    return str(path)


def resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    """线性插值重采样。做实验够用，真上生产请换 kaldi/soxr。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if sr == target or x.size == 0:
        return x
    duration = x.size / float(sr)
    out_size = int(round(duration * target))
    old_index = np.arange(x.size, dtype=np.float64) / float(sr)
    new_index = np.arange(out_size, dtype=np.float64) / float(target)
    return np.interp(new_index, old_index, x).astype(np.float32)


def _hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 40,
    fmin: float = 40.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """返回 (n_mels, n_fft // 2 + 1) 的三角滤波器组。"""
    fmax = float(fmax or sr / 2.0)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    bank = np.zeros((n_mels, bins.size), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = edges[i], edges[i + 1], edges[i + 2]
        rise = (bins - left) / max(center - left, 1e-9)
        fall = (right - bins) / max(right - center, 1e-9)
        bank[i] = np.clip(np.minimum(rise, fall), 0.0, None)
    return bank


def log_mel(
    x: np.ndarray,
    sr: int = 8000,
    n_mels: int = 40,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 256,
    fmin: float = 40.0,
    fmax: Optional[float] = None,
    preemphasis: float = 0.97,
    cmvn: bool = True,
) -> np.ndarray:
    """波形 -> (帧数, n_mels) 的 log-mel 特征，默认逐句做倒谱均值方差归一化。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((0, n_mels), dtype=np.float32)

    if preemphasis:
        x = np.concatenate([x[:1], x[1:] - preemphasis * x[:-1]]).astype(np.float32)

    frame = max(int(round(sr * frame_ms / 1000.0)), 1)
    hop = max(int(round(sr * hop_ms / 1000.0)), 1)
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))

    n_frames = 1 + (x.size - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[index] * np.hanning(frame).astype(np.float32)
    power = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

    bank = mel_filterbank(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = power @ bank.T
    out = np.log(np.maximum(mel, 1e-8))

    if cmvn and out.shape[0] > 1:
        out = (out - out.mean(axis=0, keepdims=True)) / (out.std(axis=0, keepdims=True) + 1e-5)
    return out.astype(np.float32)
