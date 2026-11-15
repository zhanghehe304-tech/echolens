"""echolens.train —— 特征装载、CTC 训练循环与评测。

所有东西都在 CPU 上跑得动：示例数据 120 句、20 轮，几十秒内收敛。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .audio import log_mel, read_wav, resample
from .data import read_manifest
from .metrics import corpus_error_rate, error_rate, greedy_decode
from .model import BLANK, CTCNet, TOKENS, model_out_lengths, token_to_id

__all__ = ["load_split", "train", "evaluate", "save_checkpoint", "load_checkpoint"]


def load_split(
    manifest: str | Path,
    root: str | Path | None = None,
    sr: int = 8000,
    n_mels: int = 40,
) -> Tuple[List[np.ndarray], List[List[str]]]:
    """读 manifest，返回 (特征列表, 音素序列列表)。相对路径默认相对 manifest 所在目录。"""
    manifest = Path(manifest)
    base = Path(root) if root is not None else manifest.parent
    feats: List[np.ndarray] = []
    texts: List[List[str]] = []
    for item in read_manifest(manifest):
        audio = Path(item["audio"])
        path = audio if audio.is_absolute() else base / audio
        rate, wave = read_wav(path)
        wave = resample(wave, rate, sr)
        feats.append(log_mel(wave, sr=sr, n_mels=n_mels))
        texts.append([t for t in str(item["text"]).replace("|", " ").split() if t])
    return feats, texts


def _encode(tokens: Sequence[str]) -> List[int]:
    return [token_to_id[t] for t in tokens]


def _make_batch(indices: Sequence[int], feats: Sequence[np.ndarray], texts: Sequence[Sequence[str]]):
    lengths = [len(feats[i]) for i in indices]
    max_len = max(lengths)
    n_mels = feats[0].shape[1]
    batch = np.zeros((len(indices), max_len, n_mels), dtype=np.float32)
    for row, i in enumerate(indices):
        batch[row, : len(feats[i])] = feats[i]

    targets: List[int] = []
    target_lengths: List[int] = []
    for i in indices:
        ids = _encode(texts[i])
        targets.extend(ids)
        target_lengths.append(len(ids))

    return (
        torch.from_numpy(batch),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )


def train(
    feats: Sequence[np.ndarray],
    texts: Sequence[Sequence[str]],
    dev_feats: Sequence[np.ndarray],
    dev_texts: Sequence[Sequence[str]],
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 3e-3,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[CTCNet, List[Dict[str, float]]]:
    """训练并返回 (模型, 每轮历史)。历史里的 ``dev_per`` 是音素错误率。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CTCNet(n_mels=feats[0].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # 余弦退火到 1e-4：CTC 前期靠大步长破 blank，后期靠小步长收尾
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-4)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    order = np.arange(len(feats))
    rng = np.random.default_rng(seed)
    history: List[Dict[str, float]] = []
    started = time.time()

    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        total_loss = 0.0
        batches = 0

        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x, lengths, targets, target_lengths = _make_batch(indices, feats, texts)

            logits = model(x, lengths)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # CTC 要 (T, B, C)
            loss = criterion(log_probs, targets, model_out_lengths(lengths), target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        summary = evaluate(model, dev_feats, dev_texts)
        scheduler.step()
        record = {
            "epoch": float(epoch + 1),
            "loss": total_loss / max(batches, 1),
            "dev_per": summary["error_rate"],
            "exact_match": summary["exact_match"],
            "elapsed": time.time() - started,
        }
        history.append(record)
        if verbose:
            print(
                f"epoch {epoch + 1:>2}/{epochs}  loss {record['loss']:.4f}  "
                f"dev PER {record['dev_per']:.3f}  exact {record['exact_match']:.3f}"
            )

    return model, history


@torch.no_grad()
def evaluate(
    model: CTCNet,
    feats: Sequence[np.ndarray],
    texts: Sequence[Sequence[str]],
) -> Dict[str, object]:
    """评测并返回汇总 + 每句细节（参考、识别、对齐操作）。"""
    model.eval()
    details: List[Dict[str, object]] = []

    for feat, ref in zip(feats, texts):
        x = torch.from_numpy(feat[None, ...])
        lengths = torch.tensor([feat.shape[0]], dtype=torch.long)
        logits = model(x, lengths)
        ids = greedy_decode(logits[0].argmax(dim=-1).tolist(), blank=BLANK)
        hyp = [TOKENS[i - 1] for i in ids if 1 <= i <= len(TOKENS)]
        err, _, ops = error_rate(ref, hyp)
        details.append({"ref": list(ref), "hyp": hyp, "errors": err, "ops": ops})

    pairs = [(d["ref"], d["hyp"]) for d in details]
    summary = corpus_error_rate(pairs)
    summary["details"] = details
    return summary


def save_checkpoint(model: CTCNet, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "tokens": list(TOKENS)}, path)
    return str(path)


def load_checkpoint(path: str | Path, n_mels: int = 40) -> CTCNet:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    model = CTCNet(n_mels=n_mels)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
