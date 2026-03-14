import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Qwen.AudioLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        const nodeConfig = {
            "批量加载音频": {
                widgetName: "文件夹路径",
                apiRoute: "/qwen/browse_folder",
                btnText: "📂 浏览文件夹"
            },
            "加载音频": {
                widgetName: "文件路径",
                apiRoute: "/qwen/browse_file",
                btnText: "🎵 浏览文件"
            }
        };

        if (nodeConfig[nodeData.name]) {
            const config = nodeConfig[nodeData.name];
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // --- 1. 通用功能：浏览按钮 ---
                const pathWidget = this.widgets.find((w) => w.name === config.widgetName);
                if (pathWidget) {
                    
                    // [修改] 已移除自定义绘制逻辑，现在将显示完整路径
                    
                    // 添加浏览按钮
                    this.addWidget("button", config.btnText, null, () => {
                        api.fetchApi(config.apiRoute, { method: "POST" })
                        .then(r => r.json())
                        .then(data => { 
                            if (data.path) {
                                pathWidget.value = data.path; 
                                pathWidget.callback(data.path); 
                            }
                        })
                        .catch(e => console.error(e));
                    });
                }

                // --- 2. 专用功能：音频可视化编辑器 ---
                if (nodeData.name === "加载音频") {
                    
                    let audioBuffer = null;
                    let audioDuration = 0;
                    let dragTarget = null;
                    let isHovering = null;

                    // UI 构建
                    const container = document.createElement("div");
                    container.style.cssText = "display:flex; flex-direction:column; gap:5px; width:100%; height:100%; min-height:120px; box-sizing:border-box; padding:6px; background:#1a1a1a; border-radius:4px; border:1px solid #333;";
                    
                    const canvas = document.createElement("canvas");
                    canvas.style.cssText = "width:100%; flex-grow:1; background:#000; border-radius:3px; display:block; cursor:default;";
                    
                    const audio = document.createElement("audio");
                    audio.controls = true;
                    audio.style.cssText = "width:100%; height:32px; flex-shrink:0; display:block;";

                    container.appendChild(canvas);
                    container.appendChild(audio);
                    
                    this.addDOMWidget("audio_visualizer", "visualizer", container);

                    const formatTime = (seconds) => {
                        const m = Math.floor(seconds / 60);
                        const s = Math.floor(seconds % 60);
                        const ms = Math.floor((seconds % 1) * 10);
                        return `${m}:${s.toString().padStart(2, '0')}.${ms}`;
                    };

                    // --- 核心绘制 ---
                    const draw = () => {
                        const width = canvas.width;
                        const height = canvas.height;
                        const ctx = canvas.getContext("2d");

                        // 背景
                        ctx.fillStyle = "#111";
                        ctx.fillRect(0, 0, width, height);

                        if (!audioBuffer || audioDuration === 0) {
                            ctx.fillStyle = "#555";
                            ctx.font = "14px Arial";
                            ctx.textAlign = "center";
                            ctx.textBaseline = "middle";
                            ctx.fillText("等待音频加载...", width / 2, height / 2);
                            return;
                        }

                        // 获取参数
                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        
                        const startTime = startWidget ? startWidget.value : 0;
                        const duration = durWidget ? durWidget.value : 0;
                        
                        let endTime = (duration > 0.001) ? (startTime + duration) : audioDuration;
                        if (endTime > audioDuration) endTime = audioDuration;

                        const pxPerSec = width / audioDuration;
                        const startX = startTime * pxPerSec;
                        const endX = endTime * pxPerSec;

                        // 波形
                        const raw = audioBuffer.getChannelData(0);
                        const step = Math.ceil(raw.length / width); 
                        const amp = (height - 30) / 2;

                        ctx.beginPath();
                        ctx.strokeStyle = "#4ade80"; 
                        ctx.lineWidth = 1;
                        for (let i = 0; i < width; i++) {
                            let min = 1.0, max = -1.0;
                            for (let j = 0; j < step; j++) {
                                const idx = (i * step) + j;
                                if (idx < raw.length) {
                                    const datum = raw[idx];
                                    if (datum < min) min = datum;
                                    if (datum > max) max = datum;
                                }
                            }
                            ctx.moveTo(i, 15 + amp + (min * amp));
                            ctx.lineTo(i, 15 + amp + (max * amp));
                        }
                        ctx.stroke();

                        // 遮罩
                        ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
                        if (startX > 0) ctx.fillRect(0, 0, startX, height);
                        if (endX < width) ctx.fillRect(endX, 0, width - endX, height);

                        // 刻度
                        ctx.fillStyle = "#888";
                        ctx.font = "10px monospace";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "alphabetic";
                        ctx.strokeStyle = "#444";
                        
                        let timeStep = 1;
                        if (audioDuration > 10) timeStep = 2;
                        if (audioDuration > 30) timeStep = 5;
                        if (audioDuration > 120) timeStep = 15;
                        if (audioDuration > 600) timeStep = 60;

                        for (let t = 0; t <= audioDuration; t += timeStep) {
                            const x = t * pxPerSec;
                            ctx.beginPath();
                            ctx.moveTo(x, 0); ctx.lineTo(x, 8);
                            ctx.stroke();
                            ctx.beginPath();
                            ctx.moveTo(x, height); ctx.lineTo(x, height - 8);
                            ctx.stroke();
                            if (x < width - 10) ctx.fillText(formatTime(t), x, 20);
                        }

                        // Start 滑块
                        ctx.lineWidth = 2;
                        ctx.strokeStyle = (dragTarget === 'start' || isHovering === 'start') ? "#ffffff" : "#00ffff";
                        ctx.fillStyle = ctx.strokeStyle;
                        
                        ctx.beginPath();
                        ctx.moveTo(startX, 0); ctx.lineTo(startX, height);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(startX, 0); ctx.lineTo(startX+8, 0); ctx.lineTo(startX, 10); ctx.fill();

                        // End 滑块
                        ctx.strokeStyle = (dragTarget === 'end' || isHovering === 'end') ? "#ffffff" : "#ff0055";
                        ctx.fillStyle = ctx.strokeStyle;

                        ctx.beginPath();
                        ctx.moveTo(endX, 0); ctx.lineTo(endX, height);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(endX, height); ctx.lineTo(endX-8, height); ctx.lineTo(endX, height-10); ctx.fill();
                    };

                    // --- 播放逻辑 ---
                    
                    audio.addEventListener('play', () => {
                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        if (!startWidget || !durWidget) return;
                        const s = startWidget.value;
                        const d = durWidget.value;
                        const e = (d > 0.001) ? s + d : audio.duration;
                        if (audio.currentTime < s - 0.1 || audio.currentTime >= e - 0.1) {
                            audio.currentTime = s;
                        }
                    });

                    audio.addEventListener('timeupdate', () => {
                        if (audio.paused) return;
                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        if (!startWidget || !durWidget) return;
                        const s = startWidget.value;
                        const d = durWidget.value;
                        if (d <= 0.001) return;
                        const e = s + d;
                        if (audio.currentTime >= e) {
                            audio.pause();
                            audio.currentTime = s; 
                        }
                    });

                    // --- 交互逻辑 ---
                    
                    const getCanvasX = (e) => {
                        const rect = canvas.getBoundingClientRect();
                        const scaleX = canvas.width / rect.width; 
                        const clientX = e.clientX - rect.left;
                        return clientX * scaleX;
                    };

                    canvas.addEventListener("mousedown", (e) => {
                        if (!audioDuration) return;
                        const x = getCanvasX(e);
                        const width = canvas.width;

                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        
                        const startTime = startWidget.value;
                        const duration = durWidget.value;
                        const endTime = (duration > 0.001) ? (startTime + duration) : audioDuration;

                        const startX = (startTime / audioDuration) * width;
                        const endX = (endTime / audioDuration) * width;
                        const threshold = 20;

                        if (Math.abs(x - endX) < threshold) {
                            dragTarget = 'end';
                        } else if (Math.abs(x - startX) < threshold) {
                            dragTarget = 'start';
                        } else {
                            updateWidgets(x, 'start_jump');
                            dragTarget = 'start';
                        }
                        requestAnimationFrame(draw);
                    });

                    window.addEventListener("mousemove", (e) => {
                        if (!audioDuration) return;
                        if (dragTarget) {
                            const x = getCanvasX(e);
                            updateWidgets(x, dragTarget);
                        } 
                        
                        const rect = canvas.getBoundingClientRect();
                        const clientX = e.clientX - rect.left;
                        const clientY = e.clientY - rect.top;

                        if (clientX >= 0 && clientX <= rect.width && clientY >= 0 && clientY <= rect.height) {
                            const x = getCanvasX(e);
                            const width = canvas.width;

                            const startWidget = this.widgets.find(w => w.name === "开始时间");
                            const durWidget = this.widgets.find(w => w.name === "持续时间");
                            const startTime = startWidget.value;
                            const duration = durWidget.value;
                            const endTime = (duration > 0.001) ? (startTime + duration) : audioDuration;

                            const startX = (startTime / audioDuration) * width;
                            const endX = (endTime / audioDuration) * width;
                            const threshold = 20;

                            let nextCursor = "crosshair";
                            let nextHover = null;

                            if (Math.abs(x - endX) < threshold) {
                                nextCursor = "ew-resize";
                                nextHover = 'end';
                            } else if (Math.abs(x - startX) < threshold) {
                                nextCursor = "ew-resize";
                                nextHover = 'start';
                            }

                            canvas.style.cursor = nextCursor;
                            if (isHovering !== nextHover) {
                                isHovering = nextHover;
                                requestAnimationFrame(draw);
                            }
                        } else {
                            isHovering = null;
                            requestAnimationFrame(draw);
                        }
                    });

                    window.addEventListener("mouseup", () => { 
                        dragTarget = null; 
                        requestAnimationFrame(draw);
                    });

                    const updateWidgets = (mouseX, mode) => {
                        const width = canvas.width;
                        let safeX = Math.max(0, Math.min(mouseX, width));
                        const time = (safeX / width) * audioDuration;
                        
                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        
                        let s = startWidget.value;
                        let d = durWidget.value;
                        let e_time = (d > 0.001) ? (s + d) : audioDuration;

                        if (mode === 'start' || mode === 'start_jump') {
                            let newStart = parseFloat(time.toFixed(2));
                            if (newStart >= e_time) newStart = e_time - 0.1;
                            if (newStart < 0) newStart = 0;
                            
                            startWidget.value = newStart;
                            startWidget.callback(newStart);

                            let newDur = parseFloat((e_time - newStart).toFixed(2));
                            if (newDur < 0) newDur = 0;
                            durWidget.value = newDur;
                            durWidget.callback(newDur);

                        } else if (mode === 'end') {
                            let newEnd = parseFloat(time.toFixed(2));
                            if (newEnd <= s) newEnd = s + 0.1;
                            if (newEnd > audioDuration) newEnd = audioDuration;

                            let newDur = parseFloat((newEnd - s).toFixed(2));
                            durWidget.value = newDur;
                            durWidget.callback(newDur);
                        }
                        requestAnimationFrame(draw);
                    };

                    const resizeObserver = new ResizeObserver(entries => {
                        for (let entry of entries) {
                            const { width, height } = entry.contentRect;
                            if (width > 0 && height > 0) {
                                canvas.width = width;
                                canvas.height = Math.max(50, height - 35); 
                                requestAnimationFrame(draw);
                            }
                        }
                    });
                    resizeObserver.observe(container);

                    // --- 加载与数据逻辑 ---

                    const getAudioUrl = (inputPath) => {
                        if (!inputPath) return "";
                        let normalizedPath = inputPath.replace(/\\/g, "/");
                        const inputIndex = normalizedPath.indexOf("/input/");
                        if (inputIndex !== -1) {
                            const relativePath = normalizedPath.substring(inputIndex + 7);
                            const lastSlash = relativePath.lastIndexOf("/");
                            let subfolder = "", filename = relativePath;
                            if (lastSlash !== -1) {
                                subfolder = relativePath.substring(0, lastSlash);
                                filename = relativePath.substring(lastSlash + 1);
                            }
                            return api.api_base + `/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}`;
                        }
                        return api.api_base + `/view?filename=${encodeURIComponent(inputPath)}&type=input`;
                    };

                    const loadAudioData = async (url) => {
                        try {
                            const response = await fetch(url);
                            const buf = await response.arrayBuffer();
                            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                            audioBuffer = await audioCtx.decodeAudioData(buf);
                            audioDuration = audioBuffer.duration;
                            draw();
                        } catch (e) { console.error(e); }
                    };

                    const updatePreview = (filePath) => {
                        if (!filePath) return;
                        
                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        if (startWidget) { startWidget.value = 0; startWidget.callback(0); }
                        if (durWidget) { durWidget.value = 0; durWidget.callback(0); }

                        audioBuffer = null;
                        audioDuration = 0;
                        requestAnimationFrame(draw); 

                        const url = getAudioUrl(filePath);
                        audio.src = url + `&t=${Date.now()}`;
                        loadAudioData(url);
                    };

                    if (pathWidget) {
                        const originalCallback = pathWidget.callback;
                        pathWidget.callback = function(value) {
                            if (originalCallback) originalCallback.call(this, value);
                            updatePreview(value);
                        };
                        setTimeout(() => { if (pathWidget.value) updatePreview(pathWidget.value); }, 500);
                    }

                    const bindWidgetRedraw = (w) => {
                        if (!w) return;
                        const cb = w.callback;
                        w.callback = function(v) {
                            if (cb) cb.call(this, v);
                            requestAnimationFrame(draw);
                        };
                    };
                    bindWidgetRedraw(this.widgets.find(w => w.name === "开始时间"));
                    bindWidgetRedraw(this.widgets.find(w => w.name === "持续时间"));

                    const onExecuted = nodeType.prototype.onExecuted;
                    nodeType.prototype.onExecuted = function (message) {
                        onExecuted?.apply(this, arguments);
                        // 依然保留此处逻辑为空，确保执行后不覆盖播放源
                    };

                    setTimeout(() => { this.setSize([400, 320]); }, 50);
                }
                return r;
            };
        }
    },
});