(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var COLORS = [
    "#4b5a6b", "#58a6ff", "#79c0ff", "#3fb950", "#56d364", "#d29922",
    "#e3b341", "#a371f7", "#bc8cff", "#db6d28", "#ff7b72"
  ];

  var state = {
    meta: null,
    mode: "synth",
    picked: [],
    busy: false,
    last: null,
    asrLast: null
  };

  function setHint(text, isError) {
    var node = $("footHint");
    node.textContent = text || "";
    node.style.color = isError ? "#f85149" : "";
  }

  function bytesToB64(u8) {
    var s = "";
    var chunk = 0x8000;
    for (var i = 0; i < u8.length; i += chunk) {
      s += String.fromCharCode.apply(null, u8.subarray(i, i + chunk));
    }
    return btoa(s);
  }

  function b64ToBytes(b64) {
    var s = atob(b64);
    var u = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) { u[i] = s.charCodeAt(i); }
    return u;
  }

  function pcmToInt16(b64) {
    var bytes = b64ToBytes(b64);
    var out = new Int16Array(Math.floor(bytes.length / 2));
    for (var i = 0; i < out.length; i++) {
      out[i] = bytes[2 * i] | (bytes[2 * i + 1] << 8);
    }
    return out;
  }

  function wavDataUri(pcmB64, sr) {
    var pcm = b64ToBytes(pcmB64);
    var buf = new ArrayBuffer(44 + pcm.length);
    var view = new DataView(buf);
    var write = function (offset, text) {
      for (var i = 0; i < text.length; i++) { view.setUint8(offset + i, text.charCodeAt(i)); }
    };
    write(0, "RIFF");
    view.setUint32(4, 36 + pcm.length, true);
    write(8, "WAVE");
    write(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sr, true);
    view.setUint32(28, sr * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    write(36, "data");
    view.setUint32(40, pcm.length, true);
    var bytes = new Uint8Array(buf);
    bytes.set(pcm, 44);
    return "data:audio/wav;base64," + bytesToB64(bytes);
  }

  function meta() {
    return fetch("/api/meta").then(function (r) { return r.json(); }).then(function (data) {
      if (!data.ok) { throw new Error(data.error || "meta 读取失败"); }
      state.meta = data;
      buildControls();
    });
  }

  function syncChips() {
    var chips = $("wordChips");
    Array.prototype.forEach.call(chips.children, function (node) {
      node.classList.toggle("on", state.picked.indexOf(node.dataset.word) >= 0);
    });
  }

  function buildControls() {
    var m = state.meta;
    $("headerStat").innerHTML =
      "checkpoint <b>" + m.checkpoint + "</b> · " + (m.params / 1000).toFixed(0) + "k 参数 · " +
      m.sr / 1000 + " kHz · CPU";

    var speaker = $("speaker");
    speaker.innerHTML = "";
    m.speakers.forEach(function (s) {
      var opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.id + (s.seen ? "（训练里见过）" : "（没见过的音色）");
      speaker.appendChild(opt);
    });

    var chips = $("wordChips");
    chips.innerHTML = "";
    m.words.forEach(function (w) {
      var chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = w;
      chip.dataset.word = w;
      chip.addEventListener("click", function () {
        var at = state.picked.indexOf(w);
        if (at >= 0) { state.picked.splice(at, 1); }
        else if (state.picked.length < 2) { state.picked.push(w); }
        else { state.picked[1] = w; }
        syncChips();
      });
      chips.appendChild(chip);
    });
    state.picked = [m.words[0]];

    var snr = $("snr");
    var tiers = [["", "干净"], ["20", "20 dB"], ["10", "10 dB"], ["0", "0 dB"], ["-5", "−5 dB"]];
    snr.innerHTML = "";
    tiers.forEach(function (pair) {
      var opt = document.createElement("option");
      opt.value = pair[0];
      opt.textContent = pair[1];
      snr.appendChild(opt);
    });

    Array.prototype.forEach.call(chips.children, function (node) {
      node.classList.toggle("on", state.picked.indexOf(node.dataset.word) >= 0);
    });

    setHint("合成模式：句子由 " + m.words.length + " 个玩具词拼成，参考音素就是这些字母。", false);

    var asr = m.asr || {};
    var asrModel = $("asrModel");
    asrModel.innerHTML = "";
    (asr.models || ["small"]).forEach(function (name) {
      var opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      asrModel.appendChild(opt);
    });
    asrModel.value = asr.default_model || "small";

    var asrLang = $("asrLang");
    asrLang.innerHTML = "";
    (asr.languages || [["", "自动识别"]]).forEach(function (pair) {
      var opt = document.createElement("option");
      opt.value = pair[0];
      opt.textContent = pair[1];
      asrLang.appendChild(opt);
    });

    if (!asr.whisper) {
      var local = $("asrEngine").options[0];
      local.textContent = "本地 faster-whisper（未安装）";
      local.disabled = true;
      $("asrEngine").value = "api";
    }
    toggleAsrRows();

    try {
      var savedKey = localStorage.getItem("echolens.asrkey");
      if (savedKey) { $("asrKey").value = savedKey; }
    } catch (err) { /* 隐私模式下 localStorage 会抛，忽略 */ }

    applyQuery();
  }

  function applyQuery() {
    var query = new URLSearchParams(window.location.search);
    var mode = query.get("mode");
    if (mode && ["synth", "pcm", "asr"].indexOf(mode) >= 0) { switchMode(mode); }
    var words = query.get("w");
    if (words) {
      state.picked = words.split(",").map(function (w) { return w.trim(); }).filter(function (w) {
        return state.meta.words.indexOf(w) >= 0;
      }).slice(0, 2);
      syncChips();
    }
    if (query.get("spk")) { $("speaker").value = query.get("spk"); }
    if (query.get("seed")) { $("seed").value = query.get("seed"); }
    if (query.has("snr")) { $("snr").value = query.get("snr"); }
    if (query.has("auto") && state.picked.length) { setTimeout(doSynth, 30); }
    if (query.has("asrdemo")) {
      // 只给截图/演示用：塞一条和真实接口返回结构相同的结果，不碰上传流程
      setTimeout(function () {
        state.asrLast = {
          engine: "whisper:small",
          language: "zh",
          duration: 5.9,
          elapsed_ms: 1640.0,
          text: "今天天气不错，我们下午三点在三楼会议室开会。",
          segments: [{ start: 0.0, end: 5.36, text: "今天天气不错，我们下午三点在三楼会议室开会。" }]
        };
        renderAsr(state.asrLast);
      }, 30);
    }
  }

  function showResults() {
    var isAsr = state.mode === "asr";
    $("asrResult").hidden = !(isAsr && state.asrLast);
    $("grid").hidden = !(!isAsr && state.last);
    $("empty").hidden = isAsr ? !!state.asrLast : !!state.last;
  }

  function toggleAsrRows() {
    var engine = $("asrEngine").value;
    $("asrApiRow").hidden = engine !== "api";
    $("asrLocalRow").hidden = engine !== "whisper";
  }

  function switchMode(mode) {
    state.mode = mode;
    $("tabSynth").classList.toggle("on", mode === "synth");
    $("tabPcm").classList.toggle("on", mode === "pcm");
    $("tabAsr").classList.toggle("on", mode === "asr");
    $("synthBox").hidden = mode !== "synth";
    $("pcmBox").hidden = mode !== "pcm";
    $("asrBox").hidden = mode !== "asr";
    if (mode === "pcm") { setHint("上传模式：参考音素未知，只给识别结果。", false); }
    if (mode === "asr") { setHint("真实转写：本地 whisper 或远程接口，输出文字和每段的时间轴。", false); }

    var titles = {
      synth: ["还没有识别结果", "右边选说话人和一两个词，点「合成并识别」。想听噪声下的表现，把「背景噪声」换到 0 dB 或 −5 dB。"],
      pcm: ["还没有识别结果", "选一段音频，浏览器先重采样到 8 kHz，再交给这套 10 音素模型，看它逐帧判成什么。"],
      asr: ["还没有转写结果", "选一个录音文件，用本地 whisper 或 OpenAI 兼容接口转写，出来的是文字和每段的时间轴。"]
    };
    var text = titles[mode] || titles.synth;
    $("emptyTitle").textContent = text[0];
    $("emptyText").textContent = text[1];
    showResults();
  }

  function setBusy(flag) {
    state.busy = flag;
    ["goSynth", "goPcm", "goAsr"].forEach(function (id) { $(id).disabled = flag; });
  }

  function request(payload, url) {
    var endpoint = url || "/api/recognize";
    setBusy(true);
    return fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (r) { return r.json(); }).then(function (data) {
      setBusy(false);
      if (!data.ok) { throw new Error(data.error || "识别失败"); }
      if (endpoint === "/api/transcribe") {
        state.asrLast = data;
        renderAsr(data);
      } else {
        state.last = data;
        render(data);
      }
      return data;
    }).catch(function (err) {
      setBusy(false);
      setHint("出错：" + err.message, true);
    });
  }

  function doSynth() {
    if (state.busy || !state.meta) { return; }
    if (!state.picked.length) { setHint("先选一个词", true); return; }
    setHint("识别中…", false);
    request({
      mode: "synth",
      speaker: $("speaker").value,
      words: state.picked.slice(),
      seed: Number($("seed").value) || 0,
      snr: $("snr").value === "" ? null : Number($("snr").value)
    }).then(function () { setHint("合成句来自 16 个玩具词，参考音素可直接比对。", false); });
  }

  function doPcm() {
    if (state.busy) { return; }
    var file = $("file").files[0];
    if (!file) { setHint("先选一个音频文件", true); return; }
    setHint("解码音频…", false);
    file.arrayBuffer().then(function (buf) {
      var tmp = new (window.AudioContext || window.webkitAudioContext)();
      return tmp.decodeAudioData(buf.slice(0)).then(function (decoded) {
        tmp.close();
        var rate = 8000;
        var frames = Math.ceil(decoded.duration * rate);
        if (frames > 20 * rate) { throw new Error("音频超过 20 秒"); }
        var off = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, Math.max(frames, 1), rate);
        var src = off.createBufferSource();
        src.buffer = decoded;
        src.connect(off.destination);
        src.start();
        return off.startRendering();
      });
    }).then(function (rendered) {
      var data = rendered.getChannelData(0);
      var bytes = new Uint8Array(data.length * 2);
      for (var i = 0; i < data.length; i++) {
        var v = Math.max(-1, Math.min(1, data[i]));
        var s = v < 0 ? Math.round(v * 32768) : Math.round(v * 32767);
        bytes[2 * i] = s & 0xff;
        bytes[2 * i + 1] = (s >> 8) & 0xff;
      }
      setHint("识别中…", false);
      return request({ mode: "pcm", sr: 8000, samples: bytesToB64(bytes) });
    }).catch(function (err) {
      state.busy = false;
      $("goPcm").disabled = false;
      setHint("出错：" + err.message, true);
    });
  }

  function doAsr() {
    if (state.busy) { return; }
    var file = $("asrFile").files[0];
    if (!file) { setHint("先选一个录音文件", true); return; }
    var limit = (state.meta && state.meta.asr && state.meta.asr.max_bytes) || 20 * 1024 * 1024;
    if (file.size > limit) {
      setHint("文件太大（上限 " + Math.round(limit / 1024 / 1024) + " MB）", true);
      return;
    }

    var engine = $("asrEngine").value;
    var payload = {
      engine: engine,
      language: $("asrLang").value,
      simplified: $("asrSimp").checked,
      filename: file.name,
      audio: ""
    };
    if (engine === "whisper") {
      payload.model = $("asrModel").value;
    } else {
      payload.model = $("asrApiModel").value.trim() || "whisper-1";
      payload.base_url = $("asrBase").value.trim() || "https://api.openai.com/v1";
      payload.api_key = $("asrKey").value.trim();
    }

    setHint("读文件…", false);
    file.arrayBuffer().then(function (buf) {
      payload.audio = bytesToB64(new Uint8Array(buf));
      setHint("转写中…（本地模型第一次会下载，可能要等几分钟）", false);
      return request(payload, "/api/transcribe");
    }).then(function (data) {
      if (!data) { return; }
      setHint("完成：" + ((data.segments || []).length) + " 段 · " + data.engine, false);
    }).catch(function (err) {
      setBusy(false);
      setHint("出错：" + err.message, true);
    });
  }

  function renderAsr(res) {
    $("asrMeta").textContent = [
      res.engine,
      res.language || "自动",
      res.duration ? res.duration.toFixed(1) + " s" : null,
      "耗时 " + (res.elapsed_ms / 1000).toFixed(1) + " s"
    ].filter(Boolean).join(" · ");

    var textNode = $("asrText");
    textNode.textContent = res.text || "（没有识别出文字）";
    textNode.classList.toggle("empty", !res.text);

    var segs = $("asrSegs");
    segs.innerHTML = "";
    (res.segments || []).forEach(function (seg) {
      var row = document.createElement("div");
      row.className = "asrSeg";
      var t = document.createElement("span");
      t.className = "t";
      t.textContent = fmtTime(seg.start) + " – " + fmtTime(seg.end);
      var s = document.createElement("span");
      s.className = "s";
      s.textContent = seg.text;
      row.appendChild(t);
      row.appendChild(s);
      segs.appendChild(row);
    });

    $("asrCopyTip").textContent = "";
    showResults();
  }

  function fmtTime(seconds) {
    var total = Math.max(0, seconds || 0);
    var minutes = Math.floor(total / 60);
    var rest = total - minutes * 60;
    return (minutes < 10 ? "0" : "") + minutes + ":" + (rest < 10 ? "0" : "") + rest.toFixed(1);
  }

  function render(res) {
    $("durLabel").textContent =
      res.duration_ms.toFixed(0) + " ms · 参考 " +
      (res.ref ? res.ref.length : 0) + " 个音素";
    $("costLabel").textContent =
      "推理 " + res.elapsed_ms.toFixed(1) + " ms · " +
      res.posteriors.frames_total + " 帧";

    $("player").src = wavDataUri(res.wave, res.sr);

    drawWave(res);
    drawRuns(res);
    drawAlign(res);
    drawHeat(res);
    showResults();
  }

  function drawWave(res) {
    var canvas = $("wave");
    var ctx = canvas.getContext("2d");
    var W = canvas.width;
    var H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0d1218";
    ctx.fillRect(0, 0, W, H);

    var pcm = pcmToInt16(res.wave);
    var columns = W;
    var step = pcm.length / columns;
    var top = 38;
    var bottom = H - 14;
    var mid = (top + bottom) / 2;

    // 音素分段：浅色底 + 分界线 + 标签
    var total = Math.max(res.posteriors.frames_total, 1);
    res.spans.forEach(function (span, index) {
      var x0 = Math.floor(span.start / total * W);
      var x1 = Math.ceil((span.end + 1) / total * W);
      ctx.fillStyle = index % 2 ? "rgba(88,166,255,.10)" : "rgba(88,166,255,.045)";
      ctx.fillRect(x0, top, Math.max(x1 - x0, 1), bottom - top);
      ctx.strokeStyle = "rgba(88,166,255,.35)";
      ctx.beginPath();
      ctx.moveTo(x0 + 0.5, top);
      ctx.lineTo(x0 + 0.5, bottom);
      ctx.stroke();
      if (x1 - x0 > 14) {
        ctx.fillStyle = "#79c0ff";
        ctx.font = "600 12px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText(span.phone, (x0 + x1) / 2, top - 8);
      }
    });

    ctx.strokeStyle = "#2b3541";
    ctx.beginPath();
    ctx.moveTo(0, mid + 0.5);
    ctx.lineTo(W, mid + 0.5);
    ctx.stroke();

    ctx.strokeStyle = "rgba(123,214,255,.85)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (var x = 0; x < columns; x++) {
      var from = Math.floor(x * step);
      var to = Math.min(Math.floor((x + 1) * step) + 1, pcm.length);
      var lo = 0;
      var hi = 0;
      for (var i = from; i < to; i++) {
        var v = pcm[i] / 32768;
        if (v < lo) { lo = v; }
        if (v > hi) { hi = v; }
      }
      var y0 = mid - hi * (bottom - top) / 2;
      var y1 = mid - lo * (bottom - top) / 2;
      ctx.moveTo(x + 0.5, y0);
      ctx.lineTo(x + 0.5, Math.max(y1, y0 + 1));
    }
    ctx.stroke();
  }

  function scoreText(run) {
    if (run.per === null) { return "参考未知"; }
    var parts = ["PER <b>" + run.per.toFixed(3) + "</b>"];
    if (run.exact) { parts.push("整句全对"); }
    else { parts.push(run.errors + " 处错"); }
    return parts.join(" · ");
  }

  function drawRuns(res) {
    var box = $("runs");
    box.innerHTML = "";
    var usable = res.runs.filter(function (r) { return r.per !== null; });
    var best = usable.length
      ? usable.reduce(function (a, b) { return b.per < a.per || (b.per === a.per && b.decoder === "beam+lm") ? b : a; })
      : res.runs[1];

    res.runs.forEach(function (run) {
      var row = document.createElement("div");
      row.className = "run" + (run === best ? " best" : "");
      var name = document.createElement("div");
      name.className = "runName";
      name.textContent = run.decoder;
      var phones = document.createElement("div");
      phones.className = "runPhones";
      phones.textContent = run.phones.join(" ") || "（空）";
      var score = document.createElement("div");
      score.className = "runScore";
      score.innerHTML = scoreText(run);
      row.appendChild(name);
      row.appendChild(phones);
      row.appendChild(score);
      box.appendChild(row);
    });
  }

  function drawAlign(res) {
    var box = $("align");
    var summary = $("alignSummary");
    box.innerHTML = "";
    summary.textContent = "";
    if (!res.ref) {
      box.innerHTML = '<div class="dim">上传模式没有参考音素，所以只给识别结果。</div>';
      return;
    }
    var run = res.runs.filter(function (r) { return r.decoder === "beam"; })[0] || res.runs[0];
    var counts = { match: 0, sub: 0, del: 0, ins: 0 };
    (run.ops || []).forEach(function (op) {
      var kind = op[0];
      var cell = document.createElement("div");
      cell.className = "cell" + (kind === "match" ? "" : " " + kind);
      cell.title = kind;
      var top = document.createElement("div");
      top.className = "top";
      top.textContent = op[1] === "" ? "—" : op[1];
      var bot = document.createElement("div");
      bot.className = "bot";
      bot.textContent = op[2] === "" ? "—" : op[2];
      cell.appendChild(top);
      cell.appendChild(bot);
      box.appendChild(cell);
      counts[kind] = (counts[kind] || 0) + 1;
    });
    summary.textContent =
      "匹配 " + counts.match + " · 替换 " + counts.sub + " · 漏识 " + counts.del + " · 多识 " + counts.ins;
  }

  function drawHeat(res) {
    var canvas = $("heat");
    var ctx = canvas.getContext("2d");
    var W = canvas.width;
    var H = canvas.height;
    var tokens = res.posteriors.tokens;
    var frames = res.posteriors.frames;
    var rows = tokens.length;
    var pad = 4;
    var rowH = (H - pad) / rows;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0d1218";
    ctx.fillRect(0, 0, W, H);

    var colW = W / Math.max(frames.length, 1);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < frames.length; c++) {
        var p = frames[c][r];
        var alpha = 0.05 + Math.pow(p, 1.5) * 0.95;
        ctx.fillStyle = COLORS[r % COLORS.length];
        ctx.globalAlpha = Math.min(1, alpha);
        ctx.fillRect(c * colW, r * rowH + pad / 2, Math.max(colW, 1), rowH);
      }
    }
    ctx.globalAlpha = 1;

    var legend = $("legend");
    legend.innerHTML = "";
    tokens.forEach(function (token, index) {
      var item = document.createElement("span");
      var swatch = document.createElement("i");
      swatch.style.background = COLORS[index % COLORS.length];
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(token));
      legend.appendChild(item);
    });

    $("heatLabel").textContent =
      "每列 = " + res.posteriors.stride + " 帧；行从上到下：" + tokens.join(" ");
  }

  $("tabSynth").addEventListener("click", function () { switchMode("synth"); });
  $("tabPcm").addEventListener("click", function () { switchMode("pcm"); });
  $("tabAsr").addEventListener("click", function () { switchMode("asr"); });
  $("goSynth").addEventListener("click", doSynth);
  $("goPcm").addEventListener("click", doPcm);
  $("goAsr").addEventListener("click", doAsr);
  $("asrEngine").addEventListener("change", toggleAsrRows);
  $("asrKey").addEventListener("change", function () {
    try { localStorage.setItem("echolens.asrkey", this.value); } catch (err) { /* 忽略 */ }
  });
  $("copyAsr").addEventListener("click", function () {
    var text = (state.asrLast && state.asrLast.text) || "";
    if (!text) { return; }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        $("asrCopyTip").textContent = "已复制";
      }).catch(function () {
        $("asrCopyTip").textContent = "复制失败，手动选中即可";
      });
    } else {
      $("asrCopyTip").textContent = "浏览器不支持自动复制，手动选中即可";
    }
  });

  meta().catch(function (err) {
    $("headerStat").textContent = "服务不可用";
    setHint("读不到 /api/meta：" + err.message, true);
  });
})();
