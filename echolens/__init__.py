"""echolens —— 一条命令跑通的语音识别流水线。

    from echolens import synth_dataset, load_split, train, evaluate
"""

from .audio import log_mel, read_wav, resample, write_wav
from .data import PHONES, WORDS, synth_dataset, synth_word
from .metrics import align, corpus_error_rate, error_rate, greedy_decode
from .model import BLANK, TOKENS, CTCNet
from .report import build_report, write_report
from .train import evaluate, load_checkpoint, load_split, save_checkpoint, train

__version__ = "0.1.0"

__all__ = [
    "PHONES",
    "WORDS",
    "BLANK",
    "TOKENS",
    "CTCNet",
    "align",
    "build_report",
    "corpus_error_rate",
    "error_rate",
    "evaluate",
    "greedy_decode",
    "load_checkpoint",
    "load_split",
    "log_mel",
    "read_wav",
    "resample",
    "save_checkpoint",
    "synth_dataset",
    "synth_word",
    "train",
    "write_report",
    "write_wav",
]
