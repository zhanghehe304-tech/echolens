# echolens

语音识别的**端到端小流水线**：合成多说话人语音 → log-mel 特征 → 小型 CTC 模型 → 三档解码（贪心 / beam search / beam + 音素 bigram）→ 音素错误率（PER）、词错误率（WER）、跨说话人泛化与逐样本对齐 → 单文件 HTML 报告。

一条命令，纯 CPU，约两分钟跑完，出来一个可以贴进周报的网页报告。

![report preview](https://aka.doubaocdn.com/s/Vgc5IexKTA)

> 上图是 `echolens demo` 生成的那份 `echolens-report.html` 的截图。

## 快速开始

```bash
pip install -e .
echolens demo
```

跑完后：

- `out/data/` — 360 个 16-bit WAV 和 `train.jsonl` / `dev.jsonl` / `dev-unseen.jsonl`
- `out/echolens.pt` — checkpoint
- `out/echolens-report.html` — 报告（图片 base64 内联，单文件，直接双击打开）

默认参数：300 句训练 / 40 句开发 / 20 句"没见过的音色" / 60 轮 / 训练集 4 个说话人 / 采样率 8 kHz。

## 实测结果

默认参数，CPU 上整跑 **119 秒**，报告内 elapsed 约 1.8 分钟（纯训练）：

| 指标 | 训练里见过的音色 | 没见过的音色 (s5–s8) |
| --- | --- | --- |
| dev 音素错误率 PER | **0.036** | 0.366 |
| dev 词错误率 WER | **0.109** | 0.750 |
| dev 整句全对率 exact match | **0.850** | 0.200 |

跨说话人泛化是这一版最想量准的东西：训练只喂 s1–s4，`dev-unseen.jsonl` 里全是 s5–s8，两个集合没有交集。0.036 对 0.366 这十倍的差，就是这个玩具模型对"陌生音色"的真实代价。

## 音高增强的取舍（这张表是选参数的依据）

不用音高增强时泛化更差，加大增强又会牺牲在域内精度。三个档位各跑一次完整训练，300 句 / 60 轮：

| 训练集音高抖动 | 见过音色 PER | 没见过音色 PER |
| --- | --- | --- |
| 0.00 | 0.045 | 0.634 |
| **0.06（默认）** | **0.036** | **0.366** |
| 0.12 | 0.086 | 0.295 |

默认取 0.06：在域内比不加增强还略好（增强顺便当了正则），陌生音色的错误从 0.634 降到 0.366。再往上加到 0.12，陌生音色还能降到 0.295，但在域内已经退到 0.086——继续加注就是在拿常用场景换极端场景。

## 解码：三种方式，收益并不均匀

| 解码 | 见过音色 PER / WER / exact | 没见过音色 PER / WER |
| --- | --- | --- |
| greedy | 0.036 / 0.109 / 0.850 | — |
| beam (宽度 10) | 0.036 / 0.109 / 0.850 | 0.366 / 0.750 |
| beam + 音素 bigram（权重 0.3） | **0.027** / 0.109 / 0.850 | 0.375 / **0.714** |

三件事值得说清：

1. **beam 这一版没有收益。** 上一版（没有音高增强）时 greedy 0.050、beam 0.045，有 0.005 的差；加了增强之后模型对每一帧都更笃定，beam 和 argmax 完全重合。这不是 beam 实现退化了，而是它本来就只在模型犹豫时才有用。
2. **音素 bigram 在熟悉的音色上把 PER 从 0.036 压到 0.027**，WER 一点没动——它纠正的是音素级的接续错误（比如 a→t 这种词表里允许的接续被误认成 a→s），但修不了整词替换。
3. **同一个 bigram 在陌生音色上 PER 反而略差（0.366→0.375），WER 却从 0.750 降到 0.714。** 模型不确定时，语言模型的偏好会把一部分音素错误换成语义上更合理的整句，代价是纯音素指标。这种"指标之间互相换"的情形，报表里不并排列出来就会被误读。

`--lm-weight 0.3` 是手定的，没有在开发集上搜索过（在开发集上搜就等于偷看测试集）；换数据要自己重调。

## 它到底在做什么

1. **合成语料**（`echolens/data.py`）。10 个音素 `a i u e o k t s m r`，基频 280–1700 Hz；16 个 3–5 音素词，每句随机 1–2 个词。八个说话人各有稳定的音色参数（基频缩放 0.85–1.32、语速基线、二三次谐波权重、底噪）；训练集用前四个，后四个留给泛化测试。另有 `--f0-jitter` 在说话人之上再整体平移基频做音高增强。
2. **特征**（`echolens/audio.py`）。纯手写：`wave` 读 WAV、线性插值重采样、numpy 三角 mel 滤波器组，输出 40 维 log-mel（25 ms 窗 / 10 ms 帧移 / n_fft 256），逐句 CMVN。
3. **模型**（`echolens/model.py`）。`Conv1d(40→64)` → `Conv1d(64→64)` → BiGRU(96) → Linear(192→11)，CTC blank=0，总计约 25 万参数。
4. **训练**（`echolens/train.py`）。Adam + 余弦退火 + weight_decay + 梯度裁剪，`CTCLoss(zero_infinity=True)`，每轮报 dev PER。
5. **测评**（`echolens/metrics.py`）。三档解码共用同一套编辑距离回溯（match / sub / del / ins）；PER 按参考音素总数加权，WER 按参考词边界切分对齐结果——一个词内出现替换/漏识/多识就记这个词错。beam 能出 nbest，`PhoneLM` 用它做 rescoring。
6. **报告**（`echolens/report.py`）。指标卡、Training 曲线、Decoder comparison、Speaker generalisation、逐样本表（`reference` / `hypothesis` / `diff` / `words` / `err`）。

## 两个值得抄下来的细节

**一、stride 设错，模型只背训练集。** 第一版两段卷积都用了 stride=2，总降采样 4 倍。一个音素只有 120 ms，4 倍降采样后只剩两三帧，CTC 学不准对齐：训练集 loss 能降到 0.03，但 dev PER 卡在 0.325，exact 只有 0.233。把第二段卷积 stride 改成 1（总降采样 2 倍）之后 dev PER 直接到 0.044。`model_out_lengths()` 与 `_make_batch()` 都只做一次 `(L-1)//2+1`，两边口径必须一致：长度算错不会报错，只会不收敛。

**二、"没见过的说话人"必须是真的没见过。** 合成数据时按句轮转分配说话人，一不小心就让同一批音色同时出现在 train 和 dev，于是报表很好看、泛化能力为零。这一版把 `dev-unseen.jsonl` 单独生成，单测里断言训练用到的音色集合与它不相交——这个断言比任何指标都更能防止自欺。

## 解码方式与说话人

```bash
echolens demo --decode greedy              # 主结果换成贪心
echolens demo --decode beam+lm             # 加音素 bigram rescoring
echolens demo --beam-size 20 --lm-weight 0.5
echolens demo --f0-jitter 0.12             # 加大音高增强
echolens demo --speakers 2 --dev-unseen 0  # 只训两个音色、不测泛化
echolens eval --manifest out/data/dev-unseen.jsonl --checkpoint out/echolens.pt --decode beam+lm
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
  data.py        合成语料、八个说话人音色、音高增强、manifest 读写
  model.py       CTCNet
  train.py       训练、评测（三档解码 + WER）、checkpoint
  metrics.py     贪心解码、beam search / nbest、PhoneLM、对齐、PER / WER
  report.py      HTML 报告
  cli.py         demo / train / eval
tests/           33 个单测，覆盖数据、特征、对齐、解码、语言模型、报告、训练
examples/        调 Python API 的示例
```

## 测试

```bash
python -m unittest discover -s tests
```

35 个用例，约 4.5 秒。CI 在 Python 3.10 / 3.11 / 3.12 上跑单测，外加一次小规模 demo 冒烟并检查报告关键字。

## 边界

说清楚它不是什么：

- **只有合成语音，没有真实录音。** 音素是带谐波的稳态音，"说话人"是一组固定的谐波参数，不是真人声纹。PER 0.036 不代表任何真实场景的识别能力，别拿它和 LibriSpeech 上的结果比。
- **音色维度是谐波结构 + 基频平移，不是共振峰。** 能用来测跨说话人泛化，但和真实方言/音色相差很远。
- **语言模型只是音素 bigram。** 词表 16 个词，它压不住词级错误；没有词级 n-gram、没有前缀树、没有用外部语料训练的 LM。
- **词表 10 个音素、16 个词。** 玩具规模，用来把流水线跑通、把对齐与泛化可视化做对，不是为了刷指标。
- 没做流式/增量推理、量化、ONNX 导出。

如果拿它当"语音识别"的基线去对标 wav2vec / Whisper，那是不成立的；拿它当"一条命令能跑完、每一步中间结果都能看到、PER 与 WER 与泛化差距都对得上"的 CTC 教学/评测骨架，它是完整的。

## License

MIT
