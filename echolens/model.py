"""echolens.model —— 一个能训得动的最小 CTC 声学模型。

结构故意做小：两层卷积负责降采样（只降 2 倍），一层 BiGRU 吃时序，线性头输出到词表。
在示例数据上 CPU 训练几十秒就能到很低的音素错误率，目的是验证流水线，不是刷指标。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .data import PHONES

__all__ = ["TOKENS", "BLANK", "token_to_id", "CTCNet", "model_out_lengths"]

BLANK = 0
TOKENS: Tuple[str, ...] = tuple(PHONES)  # 音素按定义的顺序编号，稳定可复现
token_to_id: Dict[str, int] = {token: i + 1 for i, token in enumerate(TOKENS)}


class CTCNet(nn.Module):
    """(B, T, F) log-mel -> (B, T/2, 音素数 + blank) 的 logits。"""

    def __init__(self, n_mels: int = 40, hidden: int = 96, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.gru = nn.GRU(64, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * 2, len(TOKENS) + 1)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = feats.transpose(1, 2)  # (B, F, T)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, T/2, C)
        x, _ = self.gru(x)
        return self.out(self.drop(x))


def model_out_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """卷积降采样之后的时间长度，CTC 需要它来判断最短合法长度。

    只降采样一次（stride=2），因为音素本来就短：4 倍下采样后一个音素只剩两三帧，
    CTC 很难把对齐学准。
    """
    return (lengths - 1) // 2 + 1
