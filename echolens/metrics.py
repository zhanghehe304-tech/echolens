"""echolens.metrics —— 编辑距离对齐、音素错误率、词错误率，以及逐字符 diff。

解码有两档：贪心（每帧 argmax，合并重复、去 blank）和 CTC 前缀 beam search。
贪心够快，beam 能救回一部分"两帧之间换音素太早"的错误——报告里两者都报，差值就是
beam 的真实收益。

词错误率不是另起一套对齐，而是复用音素级对齐的结果：按参考的词边界切分对齐序列，
一个词里只要出现替换/漏识/多识，就算这个词错。这样 PER 和 WER 是同一套标签算出来的，
不会出现"PER 很低但 WER 很高"这种自相矛盾的报表。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "greedy_decode",
    "beam_search_decode",
    "align",
    "error_rate",
    "corpus_error_rate",
    "word_error_rate",
    "diff_ops",
]


def greedy_decode(ids: Iterable[int], blank: int = 0) -> List[int]:
    """CTC 贪心解码：合并连续重复，再删掉 blank。"""
    out: List[int] = []
    previous = None
    for raw in ids:
        token = int(raw)
        if token != previous and token != blank:
            out.append(token)
        previous = token
    return out


def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def beam_search_decode(
    frame_log_probs: np.ndarray,
    beam: int = 10,
    blank: int = 0,
    top_k: int = 4,
) -> List[int]:
    """CTC 前缀 beam search。

    ``frame_log_probs`` 是 (帧数, 类别数) 的 log 概率。每帧只展开 top_k 个类别，
    在 11 类的小词表上已经够，且把复杂度从 类别数 压到 top_k。
    返回概率最高的前缀（已去掉 blank）。
    """
    probs = np.asarray(frame_log_probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("frame_log_probs 需要是 (帧数, 类别数)")
    if probs.shape[0] == 0:
        return []

    top_k = max(1, min(int(top_k), probs.shape[1]))
    candidates = np.argpartition(-probs, top_k - 1, axis=1)[:, :top_k]

    # prefix -> (以 blank 结尾的 log 概率, 以非 blank 结尾的 log 概率)
    blank_prob: Dict[Tuple[int, ...], float] = {(): 0.0}
    nonblank_prob: Dict[Tuple[int, ...], float] = {(): -math.inf}

    for frame in range(probs.shape[0]):
        row = probs[frame]
        next_blank: Dict[Tuple[int, ...], float] = defaultdict(lambda: -math.inf)
        next_nonblank: Dict[Tuple[int, ...], float] = defaultdict(lambda: -math.inf)

        for prefix in set(blank_prob) | set(nonblank_prob):
            pb = blank_prob.get(prefix, -math.inf)
            pnb = nonblank_prob.get(prefix, -math.inf)
            total = _log_add(pb, pnb)

            next_blank[prefix] = _log_add(next_blank[prefix], total + row[blank])

            for token in candidates[frame]:
                token = int(token)
                if token == blank:
                    continue
                score = row[token]
                last = prefix[-1] if prefix else None
                if token == last:
                    # 同一个音素连着出现两帧：非 blank 直接续，blank 后才允许再补一个
                    next_nonblank[prefix] = _log_add(next_nonblank[prefix], pnb + score)
                    extended = prefix + (token,)
                    next_nonblank[extended] = _log_add(next_nonblank[extended], pb + score)
                else:
                    extended = prefix + (token,)
                    next_nonblank[extended] = _log_add(next_nonblank[extended], total + score)

        # 只保留总分最高的 beam 条
        ranked = sorted(
            set(next_blank) | set(next_nonblank),
            key=lambda p: _log_add(next_blank[p], next_nonblank[p]),
            reverse=True,
        )[:beam]
        blank_prob = {p: next_blank[p] for p in ranked}
        nonblank_prob = {p: next_nonblank[p] for p in ranked}

    best = max((set(blank_prob) | set(nonblank_prob)),
               key=lambda p: _log_add(blank_prob[p], nonblank_prob[p]))
    return list(best)


def align(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, str, str]]:
    """基于编辑距离的全局对齐，返回 (操作, 参考, 识别) 三元组列表。

    操作取值：``match`` / ``sub``（替换）/ ``del``（漏识）/ ``ins``（多识）。
    回溯时优先走 match，让对齐结果更像人眼期待的样子。
    """
    n, m = len(ref), len(hyp)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        table[i][0] = i
    for j in range(1, m + 1):
        table[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j - 1] + cost,
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
            )

    ops: List[Tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(("match" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def error_rate(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, List[Tuple[str, str, str]]]:
    """单句 (编辑距离, 参考长度, 对齐序列)。"""
    ops = align(ref, hyp)
    errors = sum(1 for op, _, _ in ops if op != "match")
    return errors, len(ref), ops


def corpus_error_rate(pairs: Iterable[Tuple[Sequence[str], Sequence[str]]]) -> Dict[str, float]:
    """语料级错误率，按参考总长度加权（不是把每句的率再平均）。"""
    errors = 0
    total = 0
    exact = 0
    count = 0
    for ref, hyp in pairs:
        err, length, _ = error_rate(ref, hyp)
        errors += err
        total += length
        exact += 1 if list(ref) == list(hyp) else 0
        count += 1
    return {
        "error_rate": (errors / total) if total else 0.0,
        "errors": float(errors),
        "ref_tokens": float(total),
        "exact_match": (exact / count) if count else 0.0,
        "samples": float(count),
    }


def word_error_rate(
    words: Sequence[str],
    ops: Sequence[Tuple[str, str, str]],
) -> Tuple[int, int, List[bool]]:
    """按参考词的音素长度切分对齐序列，返回 (错词数, 词数, 每词是否错)。

    一个词内出现替换、漏识或多识，这个词就记错；插入不消耗参考音素，归到当前词。
    传入的 ``ops`` 必须来自同一句的 :func:`align`，否则切分会错位——发现参考音素数
    对不上就直接抛错，不猜。
    """
    lengths = [len(word) for word in words]
    if sum(lengths) == 0:
        return 0, 0, []
    ref_tokens = sum(1 for op, _, _ in ops if op != "ins")
    if ref_tokens != sum(lengths):
        raise ValueError(f"对齐序列里的参考音素数 {ref_tokens} 与词长度 {sum(lengths)} 不一致")

    flags: List[bool] = []
    consumed = 0
    current_bad = False
    word_index = 0
    for op, _, _ in ops:
        if word_index >= len(lengths):
            if op != "match" and flags:
                flags[-1] = True
            continue
        if op == "ins":
            current_bad = True
            continue
        if op != "match":
            current_bad = True
        consumed += 1
        if consumed == lengths[word_index]:
            flags.append(current_bad)
            consumed = 0
            current_bad = False
            word_index += 1

    # 对齐提前结束（理论上不会发生）时，剩下的词按"没错"处理
    flags.extend([False] * (len(lengths) - len(flags)))
    return sum(1 for flag in flags if flag), len(words), flags


def diff_ops(ref: Sequence[str], hyp: Sequence[str]) -> Dict[str, float]:
    """统计替换/漏识/多识的条数，用于「错在哪一类」的汇总。"""
    counts = {"sub": 0, "del": 0, "ins": 0, "match": 0}
    for op, _, _ in align(ref, hyp):
        counts[op] += 1
    return {k: float(v) for k, v in counts.items()}
