"""echolens.metrics —— 编辑距离对齐、音素错误率、词错误率，以及逐字符 diff。

CTC 的后处理只用贪心解码：每帧取 argmax，合并重复、去掉 blank。前缀 beam search
留给后续版本，先把「对齐到哪一步、错在哪个音素」说清楚。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = ["greedy_decode", "align", "error_rate", "corpus_error_rate", "diff_ops"]


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


def diff_ops(ref: Sequence[str], hyp: Sequence[str]) -> Dict[str, float]:
    """统计替换/漏识/多识的条数，用于「错在哪一类」的汇总。"""
    counts = {"sub": 0, "del": 0, "ins": 0, "match": 0}
    for op, _, _ in align(ref, hyp):
        counts[op] += 1
    return {k: float(v) for k, v in counts.items()}
