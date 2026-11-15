"""echolens 的命令行入口：``echolens demo`` / ``train`` / ``eval``。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main"]


def _paired_eval(model, feats, texts, word_refs, decode: str, beam: int):
    """两档解码都评一遍：``decode`` 决定哪一档当主结果，另一档拿来对比。"""
    from .train import evaluate

    main_summary = evaluate(model, feats, texts, decode=decode, beam=beam, word_refs=word_refs)
    other = "greedy" if decode == "beam" else "beam"
    reference = evaluate(model, feats, texts, decode=other, beam=beam, word_refs=word_refs)
    return main_summary, reference


def _fmt_per(summary) -> str:
    per = float(summary["error_rate"])
    wer = summary.get("word_error_rate")
    wer_text = f"  WER {float(wer):.3f}" if wer is not None else ""
    return f"PER {per:.3f}{wer_text}  exact {float(summary['exact_match']):.3f}  [{summary['decoder']}]"


def _cmd_demo(args: argparse.Namespace) -> int:
    from .data import synth_dataset
    from .report import build_report, write_report
    from .train import load_split, load_words, save_checkpoint, train

    out = Path(args.out)
    train_manifest, dev_manifest = synth_dataset(
        out / "data",
        n_train=args.n_train,
        n_dev=args.n_dev,
        seed=args.seed,
        sr=args.sr,
        n_speakers=args.speakers,
    )
    if not args.quiet:
        print(f"合成数据: {train_manifest} / {dev_manifest}")

    feats, texts = load_split(train_manifest, sr=args.sr)
    dev_feats, dev_texts = load_split(dev_manifest, sr=args.sr)
    dev_words = load_words(dev_manifest)

    model, history = train(
        feats, texts, dev_feats, dev_texts, epochs=args.epochs, seed=args.seed, verbose=not args.quiet
    )
    summary, reference = _paired_eval(model, dev_feats, dev_texts, dev_words, args.decode, args.beam_size)

    ckpt = save_checkpoint(model, out / "echolens.pt")
    report = build_report(
        summary,
        history,
        title="echolens · synthetic phones",
        notes=(
            f"CTC 模型，{len(feats)} 句训练 / {len(dev_feats)} 句开发，"
            f"{args.speakers} 个合成说话人，特征 40 维 log-mel @ 8 kHz"
        ),
        reference_run=reference,
    )
    path = write_report(report, out, name="echolens")
    if not args.quiet:
        print(f"主结果:  {_fmt_per(summary)}")
        print(f"对照:    {_fmt_per(reference)}")
        print(f"checkpoint: {ckpt}")
        print(f"report: {path}")
    else:
        print(path)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .report import build_report, write_report
    from .train import load_split, load_words, save_checkpoint, train

    feats, texts = load_split(args.manifest, root=args.root, sr=args.sr)
    if not args.dev:
        raise SystemExit("训练需要 --dev 评估集，才能看到 PER 曲线")
    dev_feats, dev_texts = load_split(args.dev, root=args.root, sr=args.sr)
    dev_words = load_words(args.dev)

    model, history = train(
        feats, texts, dev_feats, dev_texts, epochs=args.epochs, seed=args.seed, verbose=not args.quiet
    )
    summary, reference = _paired_eval(model, dev_feats, dev_texts, dev_words, args.decode, args.beam_size)
    ckpt = save_checkpoint(model, Path(args.out) / "echolens.pt")
    report = build_report(
        summary, history, title=Path(args.manifest).stem, notes=args.notes, reference_run=reference
    )
    path = write_report(report, args.out, name="echolens")
    print(f"dev {_fmt_per(summary)}")
    print(f"checkpoint: {ckpt}")
    print(f"report: {path}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .report import build_report, write_report
    from .train import load_checkpoint, load_split, load_words

    feats, texts = load_split(args.manifest, root=args.root, sr=args.sr)
    word_refs = load_words(args.manifest)
    model = load_checkpoint(args.checkpoint)
    summary, reference = _paired_eval(model, feats, texts, word_refs, args.decode, args.beam_size)
    path = write_report(
        build_report(
            summary,
            [],
            title=Path(args.manifest).stem,
            notes=args.notes,
            reference_run=reference,
        ),
        args.out,
        name="echolens",
    )
    print(_fmt_per(summary))
    print(f"对照: {_fmt_per(reference)}")
    print(f"report: {path}")
    return 0


def _add_decode_args(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--decode", choices=["greedy", "beam"], default=default, help="主结果用哪种解码")
    parser.add_argument("--beam-size", type=int, default=10, help="beam search 的宽度")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="echolens", description="语音识别流水线：合成数据 / 训练 / 评测 / 报告")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="合成数据 + 训练 + 评测 + 报告，一条命令跑完")
    demo.add_argument("--out", default="out")
    demo.add_argument("--epochs", type=int, default=60)
    demo.add_argument("--n-train", type=int, default=300)
    demo.add_argument("--n-dev", type=int, default=40)
    demo.add_argument("--sr", type=int, default=8000)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--speakers", type=int, default=4, help="合成说话人数量（最多 4）")
    _add_decode_args(demo, "beam")
    demo.add_argument("--quiet", action="store_true")
    demo.set_defaults(func=_cmd_demo)

    train_cmd = sub.add_parser("train", help="在自己的 manifest 上训练")
    train_cmd.add_argument("--manifest", required=True)
    train_cmd.add_argument("--dev", required=True)
    train_cmd.add_argument("--root", default=None, help="音频根目录，默认取 manifest 所在目录")
    train_cmd.add_argument("--out", default="out")
    train_cmd.add_argument("--epochs", type=int, default=20)
    train_cmd.add_argument("--sr", type=int, default=8000)
    train_cmd.add_argument("--seed", type=int, default=0)
    train_cmd.add_argument("--notes", default="")
    _add_decode_args(train_cmd, "greedy")
    train_cmd.add_argument("--quiet", action="store_true")
    train_cmd.set_defaults(func=_cmd_train)

    eval_cmd = sub.add_parser("eval", help="用已有 checkpoint 评测并出报告")
    eval_cmd.add_argument("--manifest", required=True)
    eval_cmd.add_argument("--checkpoint", required=True)
    eval_cmd.add_argument("--root", default=None)
    eval_cmd.add_argument("--out", default="out")
    eval_cmd.add_argument("--sr", type=int, default=8000)
    eval_cmd.add_argument("--notes", default="")
    _add_decode_args(eval_cmd, "greedy")
    eval_cmd.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
