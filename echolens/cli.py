"""echolens 的命令行入口：``echolens demo`` / ``train`` / ``eval``。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main"]


def _cmd_demo(args: argparse.Namespace) -> int:
    from .data import synth_dataset
    from .report import build_report, write_report
    from .train import evaluate, load_split, save_checkpoint, train

    out = Path(args.out)
    train_manifest, dev_manifest = synth_dataset(
        out / "data", n_train=args.n_train, n_dev=args.n_dev, seed=args.seed, sr=args.sr
    )
    if not args.quiet:
        print(f"合成数据: {train_manifest} / {dev_manifest}")

    feats, texts = load_split(train_manifest, sr=args.sr)
    dev_feats, dev_texts = load_split(dev_manifest, sr=args.sr)

    model, history = train(
        feats, texts, dev_feats, dev_texts, epochs=args.epochs, seed=args.seed, verbose=not args.quiet
    )
    summary = evaluate(model, dev_feats, dev_texts)

    ckpt = save_checkpoint(model, out / "echolens.pt")
    report = build_report(
        summary,
        history,
        title="echolens · synthetic phones",
        notes=f"CTC 模型，{len(feats)} 句训练 / {len(dev_feats)} 句开发，特征 40 维 log-mel @ 8 kHz",
    )
    path = write_report(report, out, name="echolens")
    if not args.quiet:
        print(f"dev PER {float(summary['error_rate']):.3f}  exact {float(summary['exact_match']):.3f}")
        print(f"checkpoint: {ckpt}")
        print(f"report: {path}")
    else:
        print(path)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .report import build_report, write_report
    from .train import evaluate, load_split, save_checkpoint, train

    feats, texts = load_split(args.manifest, root=args.root, sr=args.sr)
    if not args.dev:
        raise SystemExit("训练需要 --dev 评估集，才能看到 PER 曲线")
    dev_feats, dev_texts = load_split(args.dev, root=args.root, sr=args.sr)

    model, history = train(
        feats, texts, dev_feats, dev_texts, epochs=args.epochs, seed=args.seed, verbose=not args.quiet
    )
    summary = evaluate(model, dev_feats, dev_texts)
    ckpt = save_checkpoint(model, Path(args.out) / "echolens.pt")
    report = build_report(summary, history, title=Path(args.manifest).stem, notes=args.notes)
    path = write_report(report, args.out, name="echolens")
    print(f"checkpoint: {ckpt}")
    print(f"report: {path}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .report import build_report, write_report
    from .train import evaluate, load_checkpoint, load_split

    feats, texts = load_split(args.manifest, root=args.root, sr=args.sr)
    model = load_checkpoint(args.checkpoint)
    summary = evaluate(model, feats, texts)
    path = write_report(
        build_report(summary, [], title=Path(args.manifest).stem, notes=args.notes), args.out, name="echolens"
    )
    print(f"PER {float(summary['error_rate']):.3f}  exact {float(summary['exact_match']):.3f}")
    print(f"report: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="echolens", description="语音识别流水线：合成数据 / 训练 / 评测 / 报告")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="合成数据 + 训练 + 评测 + 报告，一条命令跑完")
    demo.add_argument("--out", default="out")
    demo.add_argument("--epochs", type=int, default=25)
    demo.add_argument("--n-train", type=int, default=300)
    demo.add_argument("--n-dev", type=int, default=40)
    demo.add_argument("--sr", type=int, default=8000)
    demo.add_argument("--seed", type=int, default=0)
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
    train_cmd.add_argument("--quiet", action="store_true")
    train_cmd.set_defaults(func=_cmd_train)

    eval_cmd = sub.add_parser("eval", help="用已有 checkpoint 评测并出报告")
    eval_cmd.add_argument("--manifest", required=True)
    eval_cmd.add_argument("--checkpoint", required=True)
    eval_cmd.add_argument("--root", default=None)
    eval_cmd.add_argument("--out", default="out")
    eval_cmd.add_argument("--sr", type=int, default=8000)
    eval_cmd.add_argument("--notes", default="")
    eval_cmd.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
