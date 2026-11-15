"""echolens.data —— 可控的源-滤波器合成语音，让整条流水线不依赖外部下载。

这里不再用"每个音素一个正弦"的假做法：音素的身份由**共振峰**决定。声源是
有周期性的声门脉冲串（浊音）或带通白噪声（擦音/爆破），声道是一组两极点共振峰
滤波器（F1/F2/F3），先级联再把结果过一次差分（辐射效应）。这样同一句话换个说话人，
变的是声道长度和基频，而不是把音色当随机噪声。

为什么值得这么改：v0.3.0 的"跨说话人泛化"实验里，陌生音色掉到 PER 0.634，但没人能
说清这究竟是模型学不到共性，还是合成器把音色做成了和音素无关的谐波叠加。换成
源-滤波器之后，说话人差异落在物理上有意义的地方（共振峰缩放、基频），这个指标才
开始能解释。

说话人不是一个随机种子，而是一组稳定的声学参数：基频缩放 ``f0``、声道长度缩放
``vtl``、语速基线 ``rate``、声门谱倾斜 ``tilt``、底噪 ``noise``。

换成真数据时不用改代码：准备一个 jsonl，每行
``{"audio": "a.wav", "text": "k a t o", "words": ["kato"]}``
就是 ``echolens train --manifest`` 的输入。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio import write_wav

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

# 音素 -> 发音配置。元音给三个共振峰（典型成年男声，Hz）；鼻音/流音给低位共振峰并压低增益；
# 擦音是一次高带噪声；爆破音是一段闭合加一次衰减的噪声爆破。
# 键的顺序决定 TOKENS 的编号，别随意改顺序。
PHONES: Dict[str, Dict[str, object]] = {
    "a": {"kind": "vowel", "f": (760.0, 1180.0, 2600.0), "gain": 1.0},
    "i": {"kind": "vowel", "f": (300.0, 2300.0, 2900.0), "gain": 0.95},
    "u": {"kind": "vowel", "f": (330.0, 900.0, 2300.0), "gain": 0.95},
    "e": {"kind": "vowel", "f": (480.0, 1800.0, 2600.0), "gain": 0.95},
    "o": {"kind": "vowel", "f": (500.0, 950.0, 2400.0), "gain": 0.95},
    "k": {"kind": "plosive", "f": (2200.0,), "gain": 0.75},
    "t": {"kind": "plosive", "f": (3200.0,), "gain": 0.75},
    "s": {"kind": "fricative", "f": (3400.0,), "gain": 0.65},
    "m": {"kind": "nasal", "f": (280.0, 1100.0, 2400.0), "gain": 0.55},
    "r": {"kind": "liquid", "f": (400.0, 1300.0, 2300.0), "gain": 0.8},
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

# 八个说话人。f0 是基频缩放（基准见 F0_BASE），vtl 是声道长度倍数（越长共振峰越低），
# rate 是语速基线，tilt 是声门谱倾斜（1 表示更低沉、口型更松），noise 是底噪。
# 前四个（s1–s4）默认进训练集，后四个（s5–s8）默认当"没见过的音色"做泛化测试。
SPEAKERS: Tuple[str, ...] = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")

VOICES: Dict[str, Dict[str, float]] = {
    "s1": {"f0": 0.90, "vtl": 1.06, "rate": 1.05, "tilt": 0.45, "noise": 0.030},
    "s2": {"f0": 1.35, "vtl": 0.92, "rate": 0.95, "tilt": 0.20, "noise": 0.045},
    "s3": {"f0": 1.05, "vtl": 1.00, "rate": 1.12, "tilt": 0.35, "noise": 0.038},
    "s4": {"f0": 1.50, "vtl": 0.88, "rate": 0.88, "tilt": 0.12, "noise": 0.050},
    "s5": {"f0": 1.18, "vtl": 1.10, "rate": 1.00, "tilt": 0.55, "noise": 0.033},
    "s6": {"f0": 0.80, "vtl": 1.14, "rate": 1.18, "tilt": 0.60, "noise": 0.048},
    "s7": {"f0": 1.62, "vtl": 0.90, "rate": 1.08, "tilt": 0.28, "noise": 0.042},
    "s8": {"f0": 0.95, "vtl": 0.98, "rate": 0.92, "tilt": 0.08, "noise": 0.026},
}

# 基准基频（成年男性中位），所有说话人的 f0 都是它的倍数
F0_BASE = 115.0

# 共振峰带宽：越窄的极点越"尖"，合成出来越像稳态音；给一点带宽才有真实的模糊
_BANDWIDTHS = (90.0, 110.0, 160.0)


def _voice(speaker: Optional[str]) -> Dict[str, float]:
    if speaker is None:
        return {"f0": 1.0, "vtl": 1.0, "rate": 1.0, "tilt": 0.3, "noise": 0.04}
    if speaker not in VOICES:
        raise ValueError(f"未定义的说话人: {speaker}")
    return VOICES[speaker]


def _resonator_ir(freq: float, bandwidth: float, sr: int, max_len: int = 700) -> np.ndarray:
    """两极点共振峰的冲激响应：(1 - 2r cosω z^-1 + r^2 z^-2)^-1。

    直接算出闭式解 ``r^n sin((n+1)ω) / sinω``，再当 FIR 做卷积——不引入 scipy，
    也不用手写递归循环。
    """
    freq = float(min(freq, sr / 2.0 - 200.0))  # 别贴着奈奎斯特
    freq = float(max(freq, 80.0))
    radius = float(np.exp(-np.pi * bandwidth / sr))
    omega = 2.0 * np.pi * freq / sr
    n = np.arange(max_len, dtype=np.float64)
    h = (radius**n) * np.sin(omega * (n + 1.0)) / max(np.sin(omega), 1e-6)
    # 按包络截断到 -60 dB（不能按"首个小于阈值的采样"截：零交叉会把滤波器切成 1 ms）
    tail = int(np.ceil(np.log(1e-3) / np.log(max(radius, 1e-6))) + 2)
    h = h[: max(8, min(tail, max_len))]
    return h.astype(np.float32)


def _filter(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """级联一层共振峰：卷完截回原长度，保持音素时长不变（对齐才好算）。"""
    if x.size == 0:
        return x
    full = np.convolve(x.astype(np.float64), ir.astype(np.float64))
    return full[: x.size].astype(np.float32)


def _glottal_source(length: int, sr: int, f0: float, tilt: float, rng) -> np.ndarray:
    """声门脉冲串：每个基音周期放一个升余弦脉冲，tilt 越大脉冲越宽（频谱越平）。"""
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
    # 一点点周期抖动，避免完全周期的脉冲串在频谱上过于干净
    if length > 4:
        source *= (1.0 + 0.15 * rng.random(length)).astype(np.float32)
    return source


def _noise_source(length: int, rng) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=length).astype(np.float32)


def _render_phone(
    phone: str,
    length: int,
    sr: int,
    f0: float,
    vtl: float,
    tilt: float,
    rng,
) -> np.ndarray:
    """渲染单个音素：声源 -> 共振峰级联 -> 辐射差分。返回长度恰为 ``length`` 的波形。"""
    spec = PHONES[phone]
    kind = str(spec["kind"])
    # vtl 是声道长度倍数：声道越长，共振峰越低，所以这里用 1/vtl 缩放
    formants = [float(f) / max(vtl, 1e-3) for f in spec["f"]]  # type: ignore[index]
    gain = float(spec["gain"])  # type: ignore[arg-type]

    if kind == "fricative":
        source = _noise_source(length, rng)
        for i, freq in enumerate(formants):
            source = _filter(source, _resonator_ir(freq, 700.0 + 100.0 * i, sr))
    elif kind == "plosive":
        # 前半段闭合（近似静音），后半段衰减的噪声爆破
        closure = int(length * 0.45)
        burst_len = length - closure
        source = np.zeros(length, dtype=np.float32)
        burst = _noise_source(burst_len, rng)
        for freq in formants:
            burst = _filter(burst, _resonator_ir(freq, 600.0, sr))
        decay = np.exp(-np.arange(burst_len, dtype=np.float32) / max(burst_len / 2.5, 1.0))
        source[closure:] = burst * decay
    else:
        source = _glottal_source(length, sr, f0, tilt, rng)
        if kind == "nasal":
            source = _filter(source, _resonator_ir(formants[0], 100.0, sr))  # 鼻音的低位极强
        for i, freq in enumerate(formants):
            source = _filter(source, _resonator_ir(freq, _BANDWIDTHS[min(i, len(_BANDWIDTHS) - 1)], sr))

    # 辐射效应：声道输出再过一次一阶差分，把高频抬起来（真实语音的 lips radiation）
    radiated = np.diff(source, prepend=source[:1]).astype(np.float32)
    peak = float(np.abs(radiated).max())
    if peak > 1e-6:
        radiated = radiated / peak * gain
    return radiated


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

    ``speaker`` 给定后，基频缩放、声道长度、声门倾斜、语速基线、底噪都跟着这个说话人走；
    同一个说话人 + 同一个 seed 出来的波形逐采样一致。

    ``f0_jitter`` 在说话人音色之上再整体平移基频（±f0_jitter）。现在音素身份由共振峰决定，
    所以抖基频不会改变"说了什么"，只改变"谁在说"——这正是把它当音高增强用的前提。
    """
    voice = _voice(speaker)
    rng = np.random.default_rng(seed)
    rate = float(rng.uniform(0.9, 1.15)) * voice["rate"]  # 语速
    amp = float(rng.uniform(0.6, 1.0))  # 音量
    level = voice["noise"] if noise is None else float(noise)
    vtl = voice["vtl"]
    tilt = voice["tilt"]
    f0_scale = voice["f0"] * (1.0 + float(rng.uniform(-f0_jitter, f0_jitter)))
    f0 = F0_BASE * f0_scale

    pieces: List[np.ndarray] = [np.zeros(int(sr * 0.15), dtype=np.float32)]  # 前置静音
    gap = np.zeros(int(sr * gap_ms / 1000.0 * rate), dtype=np.float32)

    for phone in word:
        if phone not in PHONES:
            raise ValueError(f"未定义的音素: {phone}")
        length = int(sr * phone_ms / 1000.0 * rate * float(rng.uniform(0.9, 1.1)))
        # 语调：每个音素上基频有 ±6% 的漂移，整句听起来不是一个死板的音高
        local_f0 = f0 * float(rng.uniform(0.94, 1.06))
        tone = _render_phone(phone, max(length, 8), sr, local_f0, vtl, tilt, rng)
        ramp = max(int(sr * 0.01), 1)
        envelope = np.ones(tone.size, dtype=np.float32)
        envelope[:ramp] = np.linspace(0.0, 1.0, min(ramp, tone.size), dtype=np.float32)
        envelope[-ramp:] = np.linspace(1.0, 0.0, min(ramp, tone.size), dtype=np.float32)
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
    bridge = np.zeros(int(sr * word_gap_ms / 1000.0), dtype=np.float32)
    for i, word in enumerate(words):
        pieces.append(
            synth_word(word, sr=sr, seed=seed + i * 17, speaker=speaker, f0_jitter=f0_jitter)
        )
        if i < len(words) - 1:
            pieces.append(bridge)
    return np.concatenate(pieces).astype(np.float32)


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

    speakers = list(SPEAKERS[: max(1, min(n_speakers, len(SPEAKERS)))]) ]
