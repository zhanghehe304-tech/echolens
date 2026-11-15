"""最小示例：不走命令行，直接调 echolens 的 Python API。

跑法（在仓库根目录）::

    python examples/train_on_your_manifest.py

大概 30 秒。想换成自己的数据，把 ``synth_dataset`` 换成 ``load_split("你的manifest.jsonl")``
就行，manifest 里每行是 ``{"audio": "相对路径.wav", "text": "k a t o"}``。
"""

from pathlib import Path

from echolens.data import synth_dataset
from echolens.report import build_report, write_report
from echolens.train import evaluate, load_split, save_checkpoint, train

out = Path("out/example")

# 1. 造数据：300 句训练 / 40 句开发，写到 out/example/data/ 下
train_manifest, dev_manifest = synth_dataset(out / "data", n_train=300, n_dev=40, seed=7)

# 2. 读音频 → log-mel 特征 + 音素序列
train_feats, train_texts = load_split(train_manifest)
dev_feats, dev_texts = load_split(dev_manifest)

# 3. 训练：前几轮几乎全输出 blank（PER 卡在 1.0），第 9 轮左右 loss 掉下来才开始会认音素，
#    所以规模小了就得多给几轮；120 句只给 20 轮的话大概停在 PER 0.7
model, history = train(train_feats, train_texts, dev_feats, dev_texts, epochs=25, seed=7)

# 4. 评测：PER 按参考音素总数加权，exact_match 是整句全对率
summary = evaluate(model, dev_feats, dev_texts)

# 5. 存 checkpoint 和报告
save_checkpoint(model, out / "echolens.pt")
report = build_report(
    summary,
    history,
    title="echolens · example",
    notes=f"{len(train_feats)} 句训练 / {len(dev_feats)} 句开发，25 轮",
)
path = write_report(report, out, name="echolens")

print(f"dev PER {float(summary['error_rate']):.3f}  exact {float(summary['exact_match']):.3f}")
print(f"report: {path}")
