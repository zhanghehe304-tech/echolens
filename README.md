# echolens

语音识别的**端到端小流水线**：合成音素语音 → log-mel 特征 → 小型 CTC 模型 → 音素错误率（PER）评测与逐样本对齐 → 单文件 HTML 报告。

一条命令，纯 CPU，约 30 秒跑完，出来一个可以贴进周报的网页报告。

![report preview](https://aka.doubaocdn.com/s/VTVXMB2Vqv)

## 快速开始

```bash
pip install -e .
echolens demo
```

跑完后：

- `out/data/` — CPU 340 个 16-bit WAV 和 `train.jsonl` / `dev.jsonl`
- `out/echolens.pt` — checkpoint
- `out/echolens-report.html` — 报告（图片 base64 内联，单文件，直接双击打开）

默认参数：300 句训练 / 40 句开发 / 25 轮 / 采样率 8 kHz。想跑快一点：

```bash
echolens demo --epochs 8 --n-train 80 --n-dev 20
```

## 实测结果

默认参数下（300 句训练、40 句开发、25 轮），CPU 上整跑 **约 31 秒**，纯训练时间约 22 秒：

| 指标 | 值 |
| --- | --- |
| dev 音素错误率 PER | **0.044** |
| dev 整句全对率 exact match | **0.825** |
| 训练用时（报告内 elapsed） | 21.7 s |

训练 25 轮 loss 从 5.2 降到 0.029，dev PER 从 1.0 降到 0.044，曲线在报告里能看到。

## 它到底在做什么

1. **合成语料**（`echolens/data.py`）。10 个音素 `a i u e o k t s m r`，基频 280–1700 Hz；16 个 3–5 音素词。每句按说话人随机语速（0.90–1.15）、音高 ±2%、音量 0.6–1.0、音素时长抖动合成，叠二次谐波和高斯噪声，落成 16-bit PCM WAV，写 JSONL manifest。
2. **特征**（`echolens/audio.py`）。纯手写：`wave` 读 WAV、线性插值重采样、numpy 三角 mel 滤波器组，输出 40 维 log-mel（25 ms 窗 / 10 ms 帧移 / n_fft 256），逐句 CMVN。
3. **模型**（`echolens/model.py`）。`Conv1d(40→64)` → `Conv1d(64→64)` → BiGRU(96) → Linear(192→11)，CTC blank=0，总计约 25 万参数。
4. **训练**（`echolens/train.py`）。Adam + 余弦退火 + weight_decay + 梯度裁剪，`CTCLoss(zero_infinity=True)`，每轮报 dev PER。
5. **评测**（`echolens/metrics.py`）。贪心解码（合并重复、删 blank），编辑距离 DP 回溯出 match / sub / del / ins，PER 按参考音素总数加权，另外统计整句全对率和最常见的替换对。
6. **报告**（`echolens/report.py`）。matplotlib Agg 出图 + base64 内联，每句一行：`reference` / `hypothesis` / `diff` / `err`，diff 里绿色是识别正确、红色是替换或漏识、蓝色是凭空多出来的音素。

## 一个值得抄下来的细节

第一版模型两段卷积都用了 stride=2，总降采样 4 倍。一个音素只有 120 ms，4 倍降采样后只剩两三帧，CTC 学不准对齐：训练集 loss 能降到 0.03，但 dev PER 卡在 0.325，exact 只有 0.233，是典型过拟合——模型把训练样本背下来了，没学到对齐。

把第二段卷积的 stride 改成 1（总降采样 2 倍）之后，dev PER 直接到 0.044，exact 0.825。`model.py` 里的 `model_out_lengths()` 与 `train.py` 里的 `_make_batch()` 都只做一次 `(L-1)//2+1`，两边口径必须一致，这是 CTC 里最容易写错又最难发现的地方：长度算错不会报错，只会不收敛。

## 用自己的数据

manifest 是每行一个 JSON：

```json
{"audio": "audio/kato-0001.wav", "text": "k a t o"}
```

相对路径按 manifest 所在目录解析，`echolens train` 可以从任意目录读：

```bash
echolens train --manifest mydata/train.jsonl --dev mydata/dev.jsonl --root mydata --epochs 20
echolens eval  --manifest mydata/dev.jsonl --checkpoint out/echolens.pt
```

`examples/train_on_your_manifest.py` 是一份 30 行、直接调 API 的版本，注释里标了每一步在干什么。

注意：模型的输出词表就是那 10 个音素 + blank。拿英文单词或中文拼音去训练它只会得到垃圾结果，想换词表得连 `PHONES` 和 `TOKENS` 一起改。

## 仓库结构

```
echolens/
  audio.py       WAV 读写、重采样、mel 滤波器组、log-mel
  data.py        合成语料、词表/音素表、manifest 读写
  model.py       CTCNet
  train.py       训练、评测、checkpoint
  metrics.py     贪心解码、对齐、PER
  report.py      HTML 报告
  cli.py         demo / train / eval
tests/           15 个单测，覆盖数据、特征、对齐、报告、训练
examples/        调 Python API 的示例
docs/images/     README 用的截图
```

## 测试

```bash
python -m unittest discover -s tests
```

15 个用例，约 4 秒。CI 在 Python 3.10 / 3.11 / 3.12 上跑单测，外加一次 3 轮的小规模 demo 冒烟。

## 边界

说清楚它不是什么：

- **只有合成语音，没有真实录音。** 声学环境干净、说话人是一段正弦叠加噪声，所以 PER 0.044 不代表任何真实场景的识别能力，别把这个数字拿去和 LibriSpeech 上的结果比。
- **词表 10 个音素、16 个词。** 这是玩具规模，用来把 CTC 流水线跑通、把对齐可视化做对，不是为了刷指标。
- **不做解码搜索。** 只有贪心解码，没有 beam search、没有前缀树、没有语言模型 rescoring。
- 没做增量推理、流式识别、量化、导出 ONNX。

如果拿它当"语音识别"的基线去对标 wav2vec / Whisper，那是不成立的；拿它当"一个能五分钟跑完、每一步都能看到中间结果的 CTC 教学/评测骨架"，它是完整的。

## License

MIT
