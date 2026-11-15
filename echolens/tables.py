"""echolens.tables —— 音素表、词表与说话人参数（纯数据，无逻辑）。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

__all__ = ["PHONES", "WORDS", "SPEAKERS", "VOICES", "F0_BASE", "BANDWIDTHS", "voice"]

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
    "kato", "saku", "tima", "suto", "kemi", "tara", "soki", "muka",
    "tesu", "kora", "sami", "toku", "mesa", "kisu", "tamo", "seta",
)

# 八个说话人。f0 是基频倍率（基准见 F0_BASE），vtl 是声道长度倍率（越长共振峰越低），
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

# 共振峰带宽：越窄的极点越"尖"；给一点带宽才有真实的模糊
BANDWIDTHS = (90.0, 110.0, 160.0)


def voice(speaker: Optional[str]) -> Dict[str, float]:
    """取说话人参数；``None`` 表示一个"标准男声"的缺省。"""
    if speaker is None:
        return {"f0": 1.0, "vtl": 1.0, "rate": 1.0, "tilt": 0.3, "noise": 0.04}
    if speaker not in VOICES:
        raise ValueError(f"未定义的说话人: {speaker}")
    return VOICES[speaker]
