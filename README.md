# echolens

语音识别的**端到端小流水线**：合成多说话人语音 → log-mel 特征 → 小型 CTC 模型 → 两档解码（贪心 / beam search）→ 音素错误率（PER）、词错误率（WER）与逐样本对齐 → 单文件 HTML 报告。

一条命令，纯 CPU，约两分钟跑完，出来一个可以贴进周报的网页报告。

![report preview](https://aka.doubaocdn.com/s/eEqevYSQd1)

> 上图是 `echolens demo` 生成的那份 `echolens-report.html` 的截图。

## 快速开始

```bash
pip install -e .
echolens demo
```

跑完后：

- `out/data/` — 340 个 16-bit WAV 和 `train.jsonl` / `dev.jsonl`
- `out/echolens.pt` — checkpoint
- `out/echolens-report.html` — 报告（图片 base64 内联，单文件，直接双击打开）

默认参数：300 句训练 / 40 句开发 / 60 轮 / 4 个合成说话人 / 采样率 8 kHz。只想确认流水线在跑：

```bash
echolens demo --epochs 6 --n-train 60 --n-dev 12
```

这时 PER 还很高（模型刚开始把帧对齐到音素），但报告、曲线、逐样本表全都是真实的，不是占位数据。

## 实测结果

默认参数（300 句训练、40 句开发、60 轮、4 说话人），CPU 上整跑 **119 秒**，报告内 elapsed 约 1.8 分钟（纯训练）：

| 指标 | beam (默认) | greedy |
| --- | --- | --- |
| dev 音素错误率 PER | **0.045** | 0.050 |
| dev 词错误率 WER | **0.164** | 0.182 |
| dev 整句全对率 exact match | **0.775** | 0.750 |

两档解码用同一个 checkpoint、同一批 40 句开发集，报告里有一张 "Decoder comparison" 表并列展示。训练 loss 从 5.2 降到 0.009。

**为什么读数比上一版（0.044）没变小。** 上一版是单说话人、每句一个词，任务简单得多；这一版有 4 个音色不同的说话人、每句 1–2 个词（6–8 个音素，词间还夹着 140 ms 停顿）。同样 300 句、25 轮时这一版只到 PER 0.386，60 轮才回到 0.050。两版数字不能直接比——训练难度变了，能比的只有同一版内部 greedy 与 beam 的差。

## 它到底在做什么

1. **合成语料**（`echolens/data.py`）。10 个音素 `a i u e o k t s m r`，基频 280–1700 Hz；16 个 3–5 音素词，每句随机 1–2 个词。四个说话人各有稳定的音色参数：基频缩放 0.90 / 1.12 / 0.98 / 1.22、语速基线、二三次谐波权重、底噪水平。句内叠随机语速（±15%）、音高抖动（±2%）、音量 0.6–1.0，落成 16-bit PCM WAV，写 JSONL manifest（带 `words` / `speaker` 字段）。
2. **特征**（`echolens/audio.py`）。纯手写：`wave` 读 WAV、线性插值重采样、numpy 三角 mel 滤波器组，输出 40 维 log-mel（25 ms 窗 / 10 ms 帧移 / n_fft 256），逐句 CMVN。
3. **模型**（`echolens/model.py`）。`Conv1d(40→64)` → `Conv1d(64→64)` → BiGRU(96) → Linear(192→11)，CTC blank=0，总计约 25 万参数。
4. **训练**（`echolens/train.py`）。Adam + 余弦退火 + weight_decay + 梯度裁剪，`CTCLoss(zero_infinity=True)`，每轮报 dev PER。
5. **测评**（`echolens/metrics.py`）。两档解码：贪心（合并重复、去 blank）与 CTC 前缀 beam search（每帧只展开 top-4，按总概率裁到 `beam` 条）。都用同一套编辑距离 DP 回溯出 match / sub / del / ins；PER 按参考音素总数加权，WER 按参考词边界切分对齐结果——一个词内出现替换/漏识/多识，就记这个词错。
6. **报告**（`echolens/report.py`）。matplotlib Agg 出图 + base64 内联；指标卡含 PER / WER / exact / 样本数 / 轮数 / 用时；Decoder comparison 表并列两档解码；每句一行 `reference` / `hypothesis` / `diff` / `words` / `err`，diff 里绿色正确、红色替换或漏识、蓝色凭空多出音素，words 列按参考词边界标出整词出错的词。

## 两个值得抄下来的细节

**一、stride 设错，模型只背训练集。** 第一版两段卷积都用了 stride=2，总降采样 4 倍。一个音素只有 120 ms，4 倍降采样后只剩两三帧，CTC 学不准对齐：训练集 loss 能降到 0.03，但 dev PER 卡在 0.325，exact 只有 0.233——模型把训练样本背下来了，没学到对齐。把第二段卷积 stride 改成 1（总降采样 2 倍）之后 dev PER 直接到 0.044。`model.py` 的 `model_out_lengths()` 与 `train.py` 的 `_make_batch()` 都只做一次 `(L-1)//2+1`，两边口径必须一致：长度算错不会报错，只会不收敛。

**二、beam 的收益取决于模型有多不确定。** 同一份代码、同一个 checkpoint：25 轮（还没收敛）时 greedy PER 0.386、beam 0.255，beam 拿回三成错误；60 轮收敛后 greedy 0.050、beam 0.045，只差一个音素级。这是 beam 的典型行为——模型每帧都很有把握时，它和 argmax 几乎重合；模型犹豫时，它才把"该选次优帧"的错误纠回来。报告里那张对比表就是给这件事留的证据，不是装饰。

## 解码方式与说话人

```bash
echolens demo --decode greedy              # 主结果换成贪心，beam 当对照
echolens demo --beam-size 20               # 加宽 beam
echolens demo --speakers 2                 # 只用两个音色
echolens eval --manifest out/data/dev.jsonl --checkpoint out/echolens.pt --decode beam
```

`--decode` 决定哪一档当"主结果"写进指标卡，另一档自动算出来放进对比表。

## 用自己的数据

manifest 是每行一个 JSON：

```json
{"audio": "audio/kato-0001.wav", "text": "k a t o", "words": ["kato"], "speaker": "s1"}
```

`words` 和 `speaker` 都是可选的。有 `words` 就能算词错误率；没有就退回 `text` 里的 `|` 分隔；连分隔符也没有，整句按一个词算——宁可给一个粗的 WER，也不假装知道词边界。相对路径按 manifest 所在目录解析：

```bash
echolens train --manifest mydata/train.jsonl --dev mydata/dev.jsonl --root mydata --epochs 20
echolens eval  --manifest mydata/dev.jsonl --checkpoint out/echolens.pt
```

`examples/train_on_your_manifest.py` 是一份直接调 Python API 的版本，注释里标了每一步在干什么。

注意：模型的输出词表就是那 10 个音素 + blank。拿英文单词或中文拼音去训练它只会得到垃圾结果，想换词表得连 `PHONES` 和 `TOKENS` 一起改。

## 仓库结构

```
echolens/
  audio.py       WAV 读写、重采样、mel 滤波器组、log-mel
  data.py        合成语料、说话人音色、manifest 读写
  model.py       CTCNet
  train.py       训练、评测（两档解码 + WER）、checkpoint
  metrics.py     贪心解码、beam search、对齐、PER / WER
  report.py      HTML 报告
  cli.py         demo / train / eval
tests/           29 个单测，覆盖数据、特征、对齐、解码、报告、训练
examples/        调 Python API 的示例
```

## 测试

```bash
python -m unittest discover -s tests
```

29 个用例，约 4 秒。CI 在 Python 3.10 / 3.11 / 3.12 上跑单测，外加一次小规模 demo 冒烟并检查报告关键字。

## 边界

说清楚它不是什么：

- **只有合成语音，没有真实录音。** 音素是带谐波的稳态音，"说话人"是一组固定的谐波参数，不是真人声纹。PER 0.045 不代表任何真实场景的识别能力，别拿它和 LibriSpeech 上的结果比。
- **音色维度是谐波结构，不是共振峰。** 四个说话人的差别落在基频、谐波权重、语速和底噪上，能用来测跨说话人泛化，但和真实方言/音色相差很远。
- **词表 10 个音素、16 个词。** 玩具规模，用来把 CTC 流水线跑通、把对齐和错误可视化做对，不是为了刷指标。
- **解码止步于前缀 beam search。** 没有语言模型、没有前缀树、没有 rescoring，也没有流式/增量推理、量化、ONNX 导出。

如果拿它当"语音识别"的基线去对标 wav2vec / Whisper，那是不成立的；拿它当"一条命令能跑完、每一步中间结果都能看到、PER 与 WER 对得上"的 CTC 教学/评测骨架，它是完整的。

## License

MIT
