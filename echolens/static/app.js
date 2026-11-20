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
    last: null
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
    applyQuery();
  }

  function applyQuery() {
    var query = new URLSearchParams(window.location.search);
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
  }

  function switchMode(mode) {
    state.mode = mode;
    $("tabSynth").classList.toggle("on", mode === "synth");
    $("tabPcm").classList.toggle("on", mode === "pcm");
    $("synthBox").hidden = mode !== "synth";
    $("pcmBox").hidden = mode !== "pcm";
    if (mode === "pcm") { setHint("上传模式：参考音素未知，只给识别结果。", false); }
  }

  function request(payload) {
    state.busy = true;
    $("goSynth").disabled = true;
    $("goPcm").disabled = true;
    return fetch("/api/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (r) { return r.json(); }).then(function (data) {
      state.busy = false;
      $("goSynth").disabled = false;
      $("goPcm").disabled = false;
      if (!data.ok) { throw new Error(data.error || "识别失败"); }
      state.last = data;
      render(data);
      return data;
    }).catch(function (err) {
      state.busy = false;
      $("goSynth").disabled = false;
      $("goPcm").disabled = false;
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

  function render(res) {
    $("empty").hidden = true;
    $("grid").hidden = false;

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
  $("goSynth").addEventListener("click", doSynth);
  $("goPcm").addEventListener("click", doPcm);

  meta().catch(function (err) {
    $("headerStat").textContent = "服务不可用";
    setHint("读不到 /api/meta：" + err.message, true);
  });
})();
