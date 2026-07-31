import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 麦克风录音节点前端扩展:
// 1. 一次性录音: 浏览器采集麦克风 -> 编码 WAV -> 上传 input 目录 -> 回填文件名
// 2. 实时翻译模式: 持续采集 -> 能量 VAD 自动断句 -> 逐句上传后端实时会话，
//    译文经 WebSocket 事件持续显示在下游实时翻译节点上 (不经过图执行)；
//    翻译节点再连“持续填充文本”节点时，每句译文自动追加到该文本框

const NODE_NAME = "Audio_Recorder";
const TRANSLATOR_NAME = "LLM_Realtime_Translator";
const ACCUMULATOR_NAME = "Text_Accumulator";

// 将 AudioBuffer 编码为 16-bit PCM WAV (浏览器 MediaRecorder 不直接产出 WAV)
function encodeWav(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const numFrames = audioBuffer.length;
    const bytesPerSample = 2;
    const dataSize = numFrames * numChannels * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
    view.setUint16(32, numChannels * bytesPerSample, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, dataSize, true);

    // 交错写入各声道采样
    const channels = [];
    for (let c = 0; c < numChannels; c++) channels.push(audioBuffer.getChannelData(c));
    let offset = 44;
    for (let i = 0; i < numFrames; i++) {
        for (let c = 0; c < numChannels; c++) {
            const s = Math.max(-1, Math.min(1, channels[c][i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
            offset += bytesPerSample;
        }
    }
    return new Blob([buffer], { type: "audio/wav" });
}

// 将 Float32 采样编码为 16-bit PCM 单声道 WAV (实时模式分段上传用)
function encodeWavFromSamples(samples, sampleRate) {
    const dataSize = samples.length * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // 单声道
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, dataSize, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        offset += 2;
    }
    return new Blob([buffer], { type: "audio/wav" });
}

app.registerExtension({
    name: "AI.AudioRecorder",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            // 隐藏"录音文件"文本框: 文件名仅作内部传参 (临时目录)，不在节点上展示，
            // 想永久保存请把"音频"输出连到 ComfyUI 原生"保存音频"节点
            const fileWidget = node.widgets && node.widgets.find((w) => w.name === "录音文件");
            if (fileWidget) {
                fileWidget.type = "hidden";
                fileWidget.computeSize = () => [0, -4];
            }

            let mediaRecorder = null;
            let mediaStream = null;
            let chunks = [];
            let recBusy = false; // 启动过程中防重复点击 (快速双击会得到空录音)

            const btn = node.addWidget("button", "🎙️ 开始录音", null, async () => {
                // ===== 停止录音 =====
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                    return;
                }
                if (recBusy) return;

                // ===== 开始录音 =====
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    alert("当前环境不支持麦克风录音 (需 localhost 或 HTTPS 访问 ComfyUI)");
                    return;
                }
                recBusy = true;
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                } catch (e) {
                    recBusy = false;
                    alert("无法访问麦克风，请检查浏览器授权: " + e.message);
                    return;
                }

                chunks = [];
                mediaRecorder = new MediaRecorder(mediaStream);
                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) chunks.push(e.data);
                };
                mediaRecorder.onstop = async () => {
                    mediaStream.getTracks().forEach((t) => t.stop());
                    btn.name = "⏳ 处理中...";
                    node.setDirtyCanvas(true, false);
                    try {
                        // 解码浏览器压缩音频并重编码为 WAV
                        const rawBlob = new Blob(chunks, { type: mediaRecorder.mimeType });
                        // 空录音 (开始后立刻停止) 无法解码，提前给出可读提示
                        if (chunks.length === 0 || rawBlob.size < 1024) {
                            throw new Error("录音时间太短，请重新录制");
                        }
                        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                        const audioBuffer = await audioCtx.decodeAudioData(await rawBlob.arrayBuffer());
                        audioCtx.close();
                        const wavBlob = encodeWav(audioBuffer);

                        // 上传到 ComfyUI input 目录
                        const formData = new FormData();
                        formData.append("audio", wavBlob, "record.wav");
                        const resp = await api.fetchApi("/qwen/upload_record", {
                            method: "POST",
                            body: formData,
                        });
                        const result = await resp.json();
                        if (result.error) throw new Error(result.error);

                        // 回填文件名
                        const pathWidget = node.widgets.find((w) => w.name === "录音文件");
                        if (pathWidget) pathWidget.value = result.filename;
                    } catch (e) {
                        alert("录音处理失败: " + e.message);
                    } finally {
                        btn.name = "🎙️ 开始录音";
                        node.setDirtyCanvas(true, false);
                    }
                };
                mediaRecorder.start();
                recBusy = false;
                btn.name = "🟥 停止录音 (录制中...)";
                node.setDirtyCanvas(true, false);
            });
            btn.serialize = false;

            // ===== 实时翻译模式: 持续采集 -> VAD 自动断句 -> 逐句上传实时会话 =====
            // VAD 参数 (参考 RealtimeSTT 等开源方案的常用默认值)
            const RMS_THRESHOLD = 0.01;  // 能量门限: 高于视为说话，低于视为静音
            const SILENCE_MS = 700;      // 静音超过此时长自动断句
            const MIN_SPEECH_MS = 300;   // 短于此时长的语音丢弃 (杂音)
            const MAX_UTTER_MS = 15000;  // 单句最长强制截断，避免长时间不出结果
            const PRE_ROLL_CHUNKS = 4;   // 句首预卷缓冲块数，防止起音被切掉
            const TAIL_KEEP_CHUNKS = 2;  // 断句后尾部保留的静音块数 (~0.17s)，其余静音尾巴裁弃

            let rt = null;      // 实时会话状态 (null = 未开启)
            let rtBusy = false; // 启动/停止过程中防重复点击

            // 沿音频输出连线查找下游实时翻译节点 (读其模型/语言参数 + 定位显示区)
            const findTranslator = () => {
                const out = node.outputs && node.outputs[0];
                if (!out || !out.links) return null;
                for (const linkId of out.links) {
                    const link = app.graph.links[linkId];
                    if (!link) continue;
                    const target = app.graph.getNodeById(link.target_id);
                    if (target && target.comfyClass === TRANSLATOR_NAME) return target;
                }
                return null;
            };

            // 沿翻译节点的译文输出连线查找“持续填充文本”节点 (可选，未连则不追加)
            const findAccumulator = (translator) => {
                const outputs = translator.outputs || [];
                for (const out of outputs) {
                    for (const linkId of out.links || []) {
                        const link = app.graph.links[linkId];
                        if (!link) continue;
                        const target = app.graph.getNodeById(link.target_id);
                        if (target && target.comfyClass === ACCUMULATOR_NAME) return target;
                    }
                }
                return null;
            };

            const setRtLabel = (label) => {
                rtBtn.name = label;
                node.setDirtyCanvas(true, false);
            };

            // 完成当前句: 拼接采样 -> 编码 WAV -> 串行上传 (保证句子顺序)
            const finalizeUtterance = () => {
                if (!rt) return;
                const chunks = rt.utter;
                const voicedMs = rt.speechMs - rt.silenceMs;
                rt.utter = [];
                rt.speaking = false;
                rt.silenceMs = 0;
                rt.speechMs = 0;
                if (voicedMs < MIN_SPEECH_MS || chunks.length === 0) return;

                // 裁掉尾部静音 (断句等待期积累的无声数据)，只留少量收尾，缩短 ASR 输入
                let end = chunks.length;
                while (end > 0 && !chunks[end - 1].voiced) end--;
                const kept = chunks.slice(0, Math.min(end + TAIL_KEEP_CHUNKS, chunks.length));

                const total = kept.reduce((s, c) => s + c.data.length, 0);
                const samples = new Float32Array(total);
                let off = 0;
                for (const c of kept) { samples.set(c.data, off); off += c.data.length; }
                const blob = encodeWavFromSamples(samples, rt.sampleRate);
                const session = rt.session;
                rt.chain = rt.chain.then(async () => {
                    try {
                        const fd = new FormData();
                        fd.append("session", session);
                        fd.append("audio", blob, "chunk.wav");
                        const resp = await api.fetchApi("/qwen/realtime/chunk", { method: "POST", body: fd });
                        const result = await resp.json();
                        if (result.error) console.warn("[实时翻译] 分段处理失败:", result.error);
                    } catch (e) {
                        console.warn("[实时翻译] 分段上传失败:", e);
                    }
                });
            };

            const startRealtime = async () => {
                const translator = findTranslator();
                if (!translator) {
                    alert("请先将本节点的\"音频\"输出连接到 🌐 实时翻译 节点");
                    return;
                }
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    alert("当前环境不支持麦克风录音 (需 localhost 或 HTTPS 访问 ComfyUI)");
                    return;
                }
                const gw = (name, dft) => {
                    const w = translator.widgets && translator.widgets.find((x) => x.name === name);
                    return w ? w.value : dft;
                };

                setRtLabel("⏳ 正在加载模型 (首次较慢，请稍候)...");
                const accumulator = findAccumulator(translator);
                let session;
                try {
                    const resp = await api.fetchApi("/qwen/realtime/start", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            node_id: String(translator.id),
                            sink_id: accumulator ? String(accumulator.id) : "",
                            model: gw("模型名称", "Hunyuan-MT-7B"),
                            target_lang: gw("目标语言", "英文"),
                            asr_model: gw("识别模型", "Qwen3-ASR-0.6B"),
                            auto_download: !!gw("自动下载模型", false),
                            max_new_tokens: gw("最大生成长度", 1024),
                        }),
                    });
                    const result = await resp.json();
                    if (result.error) throw new Error(result.error);
                    session = result.session;
                } catch (e) {
                    alert("实时翻译启动失败: " + e.message);
                    setRtLabel("🌐 开始实时翻译");
                    return;
                }

                let stream;
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        audio: { echoCancellation: true, noiseSuppression: true },
                    });
                } catch (e) {
                    alert("无法访问麦克风，请检查浏览器授权: " + e.message);
                    setRtLabel("🌐 开始实时翻译");
                    return;
                }

                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const source = ctx.createMediaStreamSource(stream);
                // ScriptProcessor 兼容性最好 (AudioWorklet 需独立模块文件，暂不引入)
                const processor = ctx.createScriptProcessor(4096, 1, 1);
                const chunkMs = (4096 / ctx.sampleRate) * 1000;

                rt = {
                    session, stream, ctx, processor,
                    sampleRate: ctx.sampleRate,
                    unload: !!gw("运行后立即卸载", true),
                    preRoll: [], utter: [],
                    speaking: false, silenceMs: 0, speechMs: 0,
                    chain: Promise.resolve(),
                };

                // 能量 VAD 状态机: 静音 -> 说话 (含预卷) -> 静音超时/超长断句
                processor.onaudioprocess = (e) => {
                    if (!rt) return;
                    const data = e.inputBuffer.getChannelData(0);
                    let sum = 0;
                    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
                    const rms = Math.sqrt(sum / data.length);
                    // 每块附带 voiced 标记，供断句时裁剪尾部静音
                    const chunk = { data: new Float32Array(data), voiced: rms >= RMS_THRESHOLD };

                    if (!rt.speaking) {
                        rt.preRoll.push(chunk);
                        if (rt.preRoll.length > PRE_ROLL_CHUNKS) rt.preRoll.shift();
                        if (chunk.voiced) {
                            rt.speaking = true;
                            rt.utter = rt.preRoll.slice();
                            rt.preRoll = [];
                            rt.speechMs = chunkMs;
                            rt.silenceMs = 0;
                        }
                    } else {
                        rt.utter.push(chunk);
                        rt.speechMs += chunkMs;
                        rt.silenceMs = chunk.voiced ? 0 : rt.silenceMs + chunkMs;
                        if (rt.silenceMs >= SILENCE_MS || rt.speechMs >= MAX_UTTER_MS) finalizeUtterance();
                    }
                };

                source.connect(processor);
                processor.connect(ctx.destination); // 部分浏览器需连接目的地才触发回调
                setRtLabel("🟢 实时翻译中 (点击停止)");
            };

            const stopRealtime = async () => {
                const cur = rt;
                if (!cur) return;
                setRtLabel("⏳ 正在结束会话...");
                if (cur.speaking) finalizeUtterance(); // 收尾最后一句
                rt = null;
                try { cur.processor.disconnect(); } catch (e) { /* 忽略 */ }
                try { cur.stream.getTracks().forEach((t) => t.stop()); } catch (e) { /* 忽略 */ }
                try { cur.ctx.close(); } catch (e) { /* 忽略 */ }
                // 等待未完成的分段避免丢尾句; 最多等 10 秒，防止分段卡住导致按钮永久失效
                await Promise.race([cur.chain, new Promise((r) => setTimeout(r, 10000))]);
                try {
                    const controller = new AbortController();
                    const timer = setTimeout(() => controller.abort(), 30000); // 后端卸载最多等 30 秒
                    await api.fetchApi("/qwen/realtime/stop", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ session: cur.session, unload: cur.unload }),
                        signal: controller.signal,
                    });
                    clearTimeout(timer);
                } catch (e) { /* 失败也无妨: 会话会自动过期，子进程有空闲超时卸载兜底 */ }
                setRtLabel("🌐 开始实时翻译");
            };

            const rtBtn = node.addWidget("button", "🌐 开始实时翻译", null, async () => {
                if (rtBusy) return;
                rtBusy = true;
                try {
                    if (rt) await stopRealtime();
                    else await startRealtime();
                } finally {
                    rtBusy = false;
                }
            });
            rtBtn.serialize = false;

            return r;
        };
    },
});
