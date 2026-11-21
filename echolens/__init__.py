"""echolens —— 一条命令跑通的语音识别流水线。

    from echolens import synth_dataset, load_split, train, evaluate
"""

from .audio import add_noise, log_mel, read_wav, resample, write_wav
from .data import F0_BASE, PHONES, SPEAKERS, VOICES, WORDS, synth_dataset, synth_sentence, synth_word
from .metrics import (
    PhoneLM,
    align,
    beam_search_decode,
    beam_search_nbest,
    corpus_error_rate,
    error_rate,
    greedy_decode,
    word_error_rate,
)
from .model import BLANK, TOKENS, CTCNet
from .report import build_report, write_report
from .train import evaluate, load_checkpoint, load_split, load_words, save_checkpoint, train

__version__ = "0.7.0"

__all__ = [
    "F0_BASE",
    "PHONES",
    "SPEAKERS",
    "VOICES",
    "WORDS",
    "BLANK",
    "TOKENS",
    "CTCNet",
    "PhoneLM",
    "add_noise",
    "align",
    "beam_search_decode",
    "beam_search_nbest",
    "build_report",
    "corpus_error_rate",
    "error_rate",
    "evaluate",
    "greedy_decode",
    "load_checkpoint",
    "load_split",
    "load_words",
    "log_mel",
    "read_wav",
    "resample",
    "save_checkpoint",
    "synth_dataset",
    "synth_sentence",
    "synth_word",
    "train",
    "word_error_rate",
    "write_report",
    "write_wav",
]
