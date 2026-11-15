"""echolens.report —— 把一次训练/评测结果渲染成单文件 HTML。

和 trainlens 的取舍一样：不发服务、不落散图，报告本身就是一个可以直接发出去的
HTML 文件。这里比那边多了「逐样本对齐」视图——错在替换、漏识还是多识，一眼能分开。
"""

from __future__ import annotations

import base64
import html
import io
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

__all__ = ["build_report", "line_chart", "diff_html"]

_STYLE = """
:root {
  --ink: #1a202c; --muted: #718096; --line: #e6e8eb;
  --bg: #f7f8fa; --accent: #2b6cb0; --bad: #c53030; --extra: #2b6cb0;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", Roboto, sans-serif;
  line-height: 1.55; -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 36px 24px 72px; }
header { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 26px 28px; }
.tag {
  display: inline-block; font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--accent); background: #ebf2fb; padding: 3px 9px; border-radius: 6px;
}
h1 { margin: 14px 0 6px; font-size: 26px; font-weight: 650; letter-spacing: -.01em; }
.meta { margin: 0; color: var(--muted); font-size: 13px; }
.notes { margin: 10px 0 0; font-size: 13px; color: #4a5568; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 24px; }
.card { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; background: #fcfcfd; }
.card .k { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.card .v { font-size: 20px; font-weight: 600; margin-top: 3px; font-variant-numeric: tabular-nums; }
section { margin-top: 30px; }
h2 { font-size: 15px; font-weight: 600; margin: 0 0 12px; padding-left: 10px; border-left: 3px solid var(--accent); }
.panel { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 18px; }
.panel img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th { text-align: left; color: var(--muted); font-weight: 500; padding: 8px; border-bottom: 1px solid var(--line); }
td { padding: 8px; border-bottom: 1px solid #f1f2f4; vertical-align: middle; }
tbody tr:last-child td { border-bottom: none; }
.note { margin: 0 0 12px; font-size: 13px; color: #4a5568; }
.mono { font-family: "Cascadia Mono", Consolas, "SF Mono", Menlo, monospace; font-size: 13px; letter-spacing: .04em; }
.ok { color: #22543d; background: #f0fff4; border-radius: 4px; padding: 1px 3px; }
.sub { color: #fff; background: var(--bad); border-radius: 4px; padding: 1px 3px; }
.del { color: var(--bad); text-decoration: line-through; background: #fff5f5; border-radius: 4px; padding: 1px 3px; }
.ins { color: #fff; background: var(--extra); border-radius: 4px; padding: 1px 3px; opacity: .85; }
.arrow { color: var(--muted); padding: 0 3px; }
footer { margin-top: 34px; color: var(--muted); font-size: 12px; text-align: center; }
"""


def _b64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def line_chart(x: Sequence[float], series: Dict[str, Sequence[float]], title: str, ylabel: str) -> str:
    """一条或多条折线，返回 base64 PNG。"""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for label, values in series.items():
        ax.plot(list(x)[: len(values)], list(values), marker="o", markersize=3.5, linewidth=1.6, label=label)
    ax.set_title(title, fontsize=11, loc="left", color="#1a202c")
    ax.set_xlabel("epoch", fontsize=9, color="#4a5568")
    ax.set_ylabel(ylabel, fontsize=9, color="#4a5568")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8, colors="#4a5568")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    return _b64(fig)


def diff_html(ops: Sequence[Tuple[str, str, str]]) -> str:
    """把对齐操作渲染成彩色片段：替换标红、漏识划线、多识标蓝。"""
    parts: List[str] = []
    for op, ref, hyp in ops:
        if op == "match":
            parts.append(f'<span class="ok">{html.escape(ref)}</span>')
        elif op == "sub":
            parts.append(
                f'<span class="sub">{html.escape(ref)}</span>'
                f'<span class="arrow">→</span>'
                f'<span class="sub">{html.escape(hyp)}</span>'
            )
        elif op == "del":
            parts.append(f'<span class="del">{html.escape(ref)}</span>')
        else:
            parts.append(f'<span class="ins">+{html.escape(hyp)}</span>')
        parts.append(" ")
    return "".join(parts).strip()


def build_report(
    summary: Dict[str, object],
    history: Sequence[Dict[str, float]],
    title: str = "echolens",
    notes: str = "",
    max_rows: int = 12,
    reference_run: Dict[str, object] | None = None,
    speaker_runs: Sequence[Tuple[str, Dict[str, object]]] | None = None,
) -> str:
    """组装 HTML 字符串。``summary`` 来自 :func:`echolens.train.evaluate`。

    ``reference_run`` 给另一档解码的评测结果时，会多出一张"解码器对比"表。
    """
    details = summary.get("details") or []
    details = list(details)

    counts = Counter()
    for row in details:
        for op, _, _ in row["ops"]:
            if op != "match":
                counts[op] += 1

    confusion = Counter()
    for row in details:
        for op, ref, hyp in row["ops"]:
            if op == "sub":
                confusion[(ref, hyp)] += 1

    # 错得最多的排在前面；全对时退化成展示前几句
    ordered = sorted(details, key=lambda row: (-int(row["errors"]), "".join(row["ref"])))
    shown = ordered[:max_rows]

    cards = [
        ("phone err rate", f"{float(summary.get('error_rate', 0.0)):.3f}"),
    ]
    if "word_error_rate" in summary:
        cards.append(("word err rate", f"{float(summary['word_error_rate']):.3f}"))
    cards += [
        ("exact match", f"{float(summary.get('exact_match', 0.0)):.3f}"),
        ("samples", str(int(summary.get("samples", 0)) or len(details))),
        ("epochs", str(len(history))),
        ("elapsed", _human(history[-1]["elapsed"]) if history else "—"),
    ]

    charts: List[str] = []
    if history:
        x = [row["epoch"] for row in history]
        charts.append(line_chart(x, {"train loss": [row["loss"] for row in history]}, "CTC loss", "loss"))
        charts.append(line_chart(x, {"dev PER": [row["dev_per"] for row in history]}, "Phone error rate", "PER"))

    has_words = any("words" in row for row in shown)

    def _word_cell(row) -> str:
        words = row.get("words") or []
        flags = row.get("word_flags") or []
        if not words:
            return ""
        return " ".join(
            f'<span class="{"sub" if bad else "ok"}">{html.escape(str(w))}</span>' for w, bad in zip(words, flags)
        )

    rows_html = "".join(
        "<tr>"
        f'<td class="mono">{html.escape(" ".join(row["ref"]))}</td>'
        f'<td class="mono">{html.escape(" ".join(row["hyp"]) or "（空）")}</td>'
        f'<td class="mono">{diff_html(row["ops"])}</td>'
        + (f'<td class="mono">{_word_cell(row)}</td>' if has_words else "")
        + f"<td>{int(row['errors'])}</td>"
        "</tr>"
        for row in shown
    )

    confusion_rows = "".join(
        f"<tr><td>{html.escape(ref)}</td><td>{html.escape(hyp)}</td><td>{n}</td></tr>"
        for (ref, hyp), n in confusion.most_common(6)
    )

    header_cells = "<th>reference</th><th>hypothesis</th><th>diff</th>"
    if has_words:
        header_cells += "<th>words</th>"
    header_cells += "<th>err</th>"

    error_panel = (
        '<div class="panel">'
        '<p class="note">按错误数从多到少排。diff 里绿色是识别正确，红色是替换或漏识，蓝色是凭空多出来的音素；'
        "words 列按参考词边界标出哪个词整词出错。</p>"
        f"<table><thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
        "</div>"
    )

    summary_panel = (
        '<div class="panel">'
        f'<p class="note">替换 {counts["sub"]} · 漏识 {counts["del"]} · 多识 {counts["ins"]}</p>'
        + (
            "<table><thead><tr><th>ref</th><th>hyp</th><th>次数</th></tr></thead>"
            f"<tbody>{confusion_rows}</tbody></table>"
            if confusion_rows
            else '<p class="note">这一轮没有被替换的音素。</p>'
        )
        + "</div>"
    )

    def _cmp_row(label: str, run: Dict[str, object]) -> str:
        wer = run.get("word_error_rate")
        wer_text = f"{float(wer):.3f}" if wer is not None else "—"
        return (
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{float(run.get('error_rate', 0.0)):.3f}</td>"
            f"<td>{wer_text}</td>"
            f"<td>{float(run.get('exact_match', 0.0)):.3f}</td></tr>"
        )

    compare_panel = ""
    if reference_run is not None:
        compare_panel = (
            '<div class="panel">'
            '<p class="note">同一个 checkpoint，只换解码方式。beam 相对 greedy 的下降就是它真实的收益。</p>'
            "<table><thead><tr><th>decoder</th><th>PER</th><th>WER</th><th>exact</th></tr></thead>"
            f"<tbody>{_cmp_row(str(reference_run.get('decoder', '?')), reference_run)}"
            f"{_cmp_row(str(summary.get('decoder', '?')), summary)}</tbody>"
            "</table>"
            "</div>"
        )

    speaker_panel = ""
    if speaker_runs:
        speaker_panel = (
            '<div class="panel">'
            '<p class="note">训练集里出现的音色 vs 从没进过训练的音色。两行的差就是跨说话人泛化的代价。</p>'
            "<table><thead><tr><th>group</th><th>PER</th><th>WER</th><th>exact</th></tr></thead>"
            "<tbody>"
            + "".join(_cmp_row(label, run) for label, run in speaker_runs)
            + "</tbody></table></div>"
        )

    body = []
    if charts:
        body.append(
            '<section><h2>Training</h2><div class="charts">'
            + "".join(f'<div class="panel"><img alt="{t}" src="data:image/png;base64,{c}"></div>' for t, c in zip(["loss", "per"], charts))
            + "</div></section>"
        )
    if reference_run is not None:
        body.append(f"<section><h2>Decoder comparison</h2>{compare_panel}</section>")

    if speaker_panel:
        body.append(f"<section><h2>Speaker generalisation</h2>{speaker_panel}</section>")

    body.append(f"<section><h2>Per-sample errors</h2>{error_panel}</section>")
    body.append(f"<section><h2>Error breakdown</h2>{summary_panel}</section>")

    cards_html = "".join(
        f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{v}</div></div>' for k, v in cards
    )
    created = datetime.now().strftime("%Y-%m-%d %H:%M")
    notes_html = f'<p class="notes">{html.escape(notes)}</p>' if notes else ""

    return (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)} · echolens report</title><style>{_STYLE}</style></head><body>"
        '<div class="wrap"><header><span class="tag">echolens report</span>'
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="meta">generated {created}</p>'
        f"{notes_html}"
        f'<div class="cards">{cards_html}</div></header>'
        + "".join(body)
        + f'<footer>Generated by echolens · {created}</footer></div></body></html>'
    )


def _human(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


def write_report(html_text: str, out_dir: str | Path, name: str = "echolens") -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}-report.html"
    path.write_text(html_text, encoding="utf-8")
    return str(path)
