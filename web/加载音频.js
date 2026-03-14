import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Qwen.AudioLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        const nodeConfig = {
            "批量加载音频": { widgetName: "文件夹路径", apiRoute: "/qwen/browse_folder", btnText: "📂 浏览文件夹" },
            "加载音频": { widgetName: "文件路径", apiRoute: "/qwen/browse_file", btnText: "🎵 浏览媒体文件" }
        };

        if (nodeData.name === "批量加载音频") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const widgetsToHide = ["文件夹路径", "文件扩展名"];
                widgetsToHide.forEach(name => {
                    const w = this.widgets.find(wg => wg.name === name);
                    if (w) {
                        if (w.inputEl) {
                            w.inputEl.style.display = "none";
                            w.inputEl.style.opacity = "0";
                        }
                        w.computeSize = () => [0, -4];
                        w.draw = () => {};
                    }
                });
                
                return r;
            };
        }

        if (nodeConfig[nodeData.name]) {
            const config = nodeConfig[nodeData.name];
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                let MIN_WIDTH = 400;
                let MIN_HEIGHT = 300; 

                if (nodeData.name === "批量加载音频") {
                    MIN_WIDTH = 320;
                    MIN_HEIGHT = 140; 
                    
                    const extWidget = this.widgets.find(wg => wg.name === "文件扩展名");
                    if (extWidget) {
                        if (extWidget.inputEl) {
                            extWidget.inputEl.style.display = "none";
                            extWidget.inputEl.style.opacity = "0";
                        }
                        extWidget.computeSize = () => [0, -4];
                        extWidget.draw = () => {};
                    }
                }

                const pathWidget = this.widgets.find((w) => w.name === config.widgetName);
                if (pathWidget) {
                    if (pathWidget.inputEl) {
                        pathWidget.inputEl.style.display = "none";
                        pathWidget.inputEl.style.opacity = "0";
                    }
                    
                    if (nodeData.name === "加载音频") {
                        pathWidget.computeSize = () => [0, -4];
                        pathWidget.draw = () => {};
                    } else if (nodeData.name === "批量加载音频") {
                        pathWidget.computeSize = function(width) {
                            return [width, 26]; 
                        };

                        pathWidget.draw = function(ctx, node, widgetWidth, y, widgetHeight) {
                            ctx.fillStyle = "#222";
                            ctx.fillRect(0, y, widgetWidth, widgetHeight);
                            ctx.strokeStyle = "#444";
                            ctx.strokeRect(0, y, widgetWidth, widgetHeight);
                            
                            const realValue = this.value || "";
                            let displayValue = "未选择文件夹";
                            if (typeof realValue === "string" && realValue.length > 0) {
                                const parts = realValue.replace(/\\/g, '/').split('/');
                                displayValue = parts.pop() || parts.pop() || realValue; 
                                
                                if (displayValue.toLowerCase() === "audio") {
                                    displayValue = "音频目录 (Audio)";
                                }
                            }
                            
                            ctx.fillStyle = "#ccc";
                            ctx.font = "12px Arial";
                            ctx.textAlign = "left";
                            ctx.textBaseline = "middle";
                            ctx.fillText("📂 " + displayValue, 8, y + widgetHeight * 0.5);
                        };
                    }

                    this.addWidget("button", config.btnText, null, () => {
                        api.fetchApi(config.apiRoute, { method: "POST" })
                        .then(r => r.json())
                        .then(data => { if (data.path) { pathWidget.value = data.path; pathWidget.callback(data.path); }})
                        .catch(e => console.error(e));
                    });
                }

                if (nodeData.name === "加载音频") {
                    let audioBuffer = null;
                    let audioDuration = 0;
                    let dragTarget = null;
                    let isHovering = null;
                    let lastLoadedPath = "";
                    let _isConfiguring = false; // 状态锁

                    const container = document.createElement("div");
                    container.style.cssText = "display:flex; flex-direction:column; gap:6px; width:100%; height:100%; box-sizing:border-box; padding:8px; background:#1a1a1a; border-radius:4px; border:1px solid #333;";
                    
                    const titleBar = document.createElement("div");
                    titleBar.style.cssText = "background:#222; color:#eee; text-align:center; padding:6px 10px; border-radius:4px; font-size:13px; font-family:sans-serif; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex-shrink:0; border:1px solid #111;";
                    titleBar.innerText = "未选择音频/视频";
                    
                    const canvas = document.createElement("canvas");
                    canvas.style.cssText = "width:100%; flex-grow:1; background:#000; border-radius:3px; display:block; cursor:default; min-height: 80px;";
                    
                    const audio = document.createElement("audio");
                    audio.controls = true;
                    audio.style.cssText = "width:100%; height:32px; flex-shrink:0; display:block;";

                    container.appendChild(titleBar);
                    container.appendChild(canvas);
                    container.appendChild(audio);
                    
                    this.addDOMWidget("audio_visualizer", "visualizer", container);

                    const formatTime = (seconds) => {
                        const m = Math.floor(seconds / 60);
                        const s = Math.floor(seconds % 60);
                        const ms = Math.floor((seconds % 1) * 10);
                        return `${m}:${s.toString().padStart(2, '0')}.${ms}`;
                    };

                    const draw = () => {
                        const width = canvas.width;
                        const height = canvas.height;
                        const ctx = canvas.getContext("2d");

                        ctx.fillStyle = "#111";
                        ctx.fillRect(0, 0, width, height);

                        if (!audioBuffer || audioDuration === 0) {
                            ctx.fillStyle = "#555";
                            ctx.font = "14px Arial";
                            ctx.textAlign = "center";
                            ctx.textBaseline = "middle";
                            ctx.fillText("等待文件解析...", width / 2, height / 2);
                            return;
                        }

                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        
                        const startTime = startWidget ? parseFloat(startWidget.value) : 0;
                        const duration = durWidget ? parseFloat(durWidget.value) : 0;
                        
                        let endTime = (duration > 0.001) ? (startTime + duration) : audioDuration;
                        if (endTime > audioDuration) endTime = audioDuration;

                        const pxPerSec = width / audioDuration;
                        const startX = startTime * pxPerSec;
                        const endX = endTime * pxPerSec;

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

                        ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
                        if (startX > 0) ctx.fillRect(0, 0, startX, height);
                        if (endX < width) ctx.fillRect(endX, 0, width - endX, height);

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
                            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 8); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(x, height); ctx.lineTo(x, height - 8); ctx.stroke();
                            if (x < width - 10) ctx.fillText(formatTime(t), x, 20);
                        }

                        ctx.lineWidth = 2;
                        ctx.strokeStyle = (dragTarget === 'start' || isHovering === 'start') ? "#ffffff" : "#00ffff";
                        ctx.fillStyle = ctx.strokeStyle;
                        
                        ctx.beginPath(); ctx.moveTo(startX, 0); ctx.lineTo(startX, height); ctx.stroke();
                        ctx.beginPath(); ctx.moveTo(startX, 0); ctx.lineTo(startX+8, 0); ctx.lineTo(startX, 10); ctx.fill();

                        ctx.strokeStyle = (dragTarget === 'end' || isHovering === 'end') ? "#ffffff" : "#ff0055";
                        ctx.fillStyle = ctx.strokeStyle;

                        ctx.beginPath(); ctx.moveTo(endX, 0); ctx.lineTo(endX, height); ctx.stroke();
                        ctx.beginPath(); ctx.moveTo(endX, height); ctx.lineTo(endX-8, height); ctx.lineTo(endX, height-10); ctx.fill();
                    };

                    audio.addEventListener('play', () => {
                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        if (!startWidget || !durWidget) return;
                        const s = parseFloat(startWidget.value) || 0;
                        const d = parseFloat(durWidget.value) || 0;
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
                        const s = parseFloat(startWidget.value) || 0;
                        const d = parseFloat(durWidget.value) || 0;
                        if (d <= 0.001) return;
                        const e = s + d;
                        if (audio.currentTime >= e) {
                            audio.pause();
                            audio.currentTime = s; 
                        }
                    });

                    const getCanvasX = (e) => {
                        const rect = canvas.getBoundingClientRect();
                        return (e.clientX - rect.left) * (canvas.width / rect.width);
                    };

                    canvas.addEventListener("mousedown", (e) => {
                        if (!audioDuration) return;
                        const x = getCanvasX(e);
                        const width = canvas.width;
                        const pxPerSec = width / audioDuration;
                        const threshold = 20;

                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        
                        const sVal = parseFloat(startWidget.value) || 0;
                        const dVal = parseFloat(durWidget.value) || 0;
                        const sX = sVal * pxPerSec;
                        const eX = (dVal > 0.001 ? sVal + dVal : audioDuration) * pxPerSec;

                        const distS = Math.abs(x - sX);
                        const distE = Math.abs(x - eX);

                        if (distS < threshold && distE < threshold) {
                            dragTarget = distS < distE ? 'start' : 'end';
                        } else if (distE < threshold) {
                            dragTarget = 'end';
                        } else if (distS < threshold) {
                            dragTarget = 'start';
                        } else {
                            dragTarget = 'start'; 
                            updateWidgets(x, 'start_jump');
                        }
                        
                        if (dragTarget === 'end') {
                            audio.currentTime = sVal + dVal;
                        } else {
                            audio.currentTime = sVal;
                        }
                        requestAnimationFrame(draw);
                    });

                    window.addEventListener("mousemove", (e) => {
                        if (!audioDuration) return;
                        if (dragTarget) {
                            const x = getCanvasX(e);
                            updateWidgets(x, dragTarget);
                            
                            const startWidget = this.widgets.find(w => w.name === "开始时间");
                            const durWidget = this.widgets.find(w => w.name === "持续时间");
                            const sVal = parseFloat(startWidget.value) || 0;
                            const dVal = parseFloat(durWidget.value) || 0;
                            
                            if (dragTarget === 'end') {
                                audio.currentTime = sVal + dVal; 
                            } else {
                                audio.currentTime = sVal;
                            }
                        } 
                        
                        const rect = canvas.getBoundingClientRect();
                        const mx = e.clientX - rect.left;
                        if (mx >= 0 && mx <= rect.width && e.clientY - rect.top >= 0 && e.clientY - rect.top <= rect.height) {
                            const x = getCanvasX(e);
                            const pxPerSec = canvas.width / audioDuration;
                            
                            const startWidget = this.widgets.find(w => w.name === "开始时间");
                            const durWidget = this.widgets.find(w => w.name === "持续时间");
                            const sVal = parseFloat(startWidget.value) || 0;
                            const dVal = parseFloat(durWidget.value) || 0;
                            
                            const sX = sVal * pxPerSec;
                            const eX = (dVal > 0.001 ? sVal + dVal : audioDuration) * pxPerSec;
                            
                            let nCursor = "crosshair", nHover = null;
                            if (Math.abs(x - eX) < 20) { nCursor = "ew-resize"; nHover = 'end'; }
                            else if (Math.abs(x - sX) < 20) { nCursor = "ew-resize"; nHover = 'start'; }
                            
                            canvas.style.cursor = nCursor;
                            if (isHovering !== nHover) { isHovering = nHover; requestAnimationFrame(draw); }
                        } else {
                            isHovering = null; requestAnimationFrame(draw);
                        }
                    });

                    window.addEventListener("mouseup", () => { 
                        if (dragTarget) {
                            const sW = this.widgets.find(w => w.name === "开始时间");
                            if (sW) audio.currentTime = parseFloat(sW.value) || 0;
                        }
                        dragTarget = null; 
                        requestAnimationFrame(draw); 
                    });

                    const updateWidgets = (mouseX, mode) => {
                        const width = canvas.width;
                        let safeX = Math.max(0, Math.min(mouseX, width));
                        const time = (safeX / width) * audioDuration;
                        
                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        
                        let s = parseFloat(startWidget.value) || 0;
                        let d = parseFloat(durWidget.value) || 0;
                        let e_time = (d > 0.001) ? (s + d) : audioDuration;

                        if (mode === 'start' || mode === 'start_jump') {
                            let nS = parseFloat(time.toFixed(2));
                            if (nS >= e_time) nS = Math.max(0, e_time - 0.05);
                            
                            startWidget.value = parseFloat(nS.toFixed(2));
                            startWidget.callback(startWidget.value);

                            let nD = parseFloat((e_time - startWidget.value).toFixed(2));
                            durWidget.value = Math.max(0, nD);
                            durWidget.callback(durWidget.value);

                        } else if (mode === 'end') {
                            let nE = parseFloat(time.toFixed(2));
                            if (nE <= s) nE = s + 0.05;

                            let nD = parseFloat((Math.min(nE, audioDuration) - s).toFixed(2));
                            durWidget.value = Math.max(0, nD);
                            durWidget.callback(durWidget.value);
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

                    const getMediaUrl = (inputPath) => {
                        if (!inputPath) return "";
                        return api.api_base + `/qwen/view_media?path=${encodeURIComponent(inputPath)}`;
                    };

                    const loadAudioData = async (url) => {
                        try {
                            const response = await fetch(url);
                            const buf = await response.arrayBuffer();
                            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                            audioBuffer = await audioCtx.decodeAudioData(buf);
                            audioDuration = audioBuffer.duration;

                            // 跳转至缓存的时间
                            const sW = this.widgets.find(w => w.name === "开始时间");
                            if (sW && parseFloat(sW.value) > 0) audio.currentTime = parseFloat(sW.value);

                            draw();
                        } catch (e) { console.error("Waveform decode failed:", e); }
                    };

                    const updatePreview = (filePath, isInit = false) => {
                        if (!filePath) {
                            titleBar.innerText = "未选择音频/视频";
                            return;
                        }
                        
                        titleBar.innerText = filePath.replace(/\\/g, '/').split('/').pop();
                        
                        if (filePath === lastLoadedPath) return; 
                        lastLoadedPath = filePath;

                        const isGraphLoading = app.configuringGraph || _isConfiguring;
                        if (!isInit && !isGraphLoading) {
                            const startWidget = this.widgets.find(w => w.name === "开始时间");
                            const durWidget = this.widgets.find(w => w.name === "持续时间");
                            if (startWidget) { startWidget.value = 0; startWidget.callback(0); }
                            if (durWidget) { durWidget.value = 0; durWidget.callback(0); }
                        }

                        audioBuffer = null;
                        audioDuration = 0;
                        requestAnimationFrame(draw); 

                        const url = getMediaUrl(filePath);
                        audio.src = url + `&t=${Date.now()}`;
                        loadAudioData(url);
                    };

                    if (pathWidget) {
                        const originalCallback = pathWidget.callback;
                        pathWidget.callback = function(value) {
                            if (originalCallback) originalCallback.call(this, value);
                            updatePreview(value, false);
                        };
                    }

                    const origOnConfigure = this.onConfigure;
                    this.onConfigure = function(info) {
                        _isConfiguring = true; // 锁定
                        if (origOnConfigure) origOnConfigure.apply(this, arguments);
                        if (pathWidget && pathWidget.value) {
                            updatePreview(pathWidget.value, true);
                        }
                        _isConfiguring = false; // 解除
                    };

                    setTimeout(() => { 
                        if (pathWidget && pathWidget.value && audioDuration === 0) {
                            updatePreview(pathWidget.value, true); 
                        }
                    }, 500);

                    this.widgets.forEach(w => {
                        if (w.name === "开始时间" || w.name === "持续时间") {
                            const cb = w.callback;
                            w.callback = function(v) {
                                if (cb) cb.call(this, v);
                                requestAnimationFrame(draw);
                            };
                        }
                    });
                }

                const origOnResize = this.onResize;
                this.onResize = function(size) {
                    if (origOnResize) origOnResize.apply(this, arguments);
                    if (size[0] < MIN_WIDTH) size[0] = MIN_WIDTH;
                    if (size[1] < MIN_HEIGHT) size[1] = MIN_HEIGHT;
                };

                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecuted?.apply(this, arguments);
                };

                // 不再强制覆盖历史保存的尺寸
                if (this.size[0] < MIN_WIDTH) this.size[0] = MIN_WIDTH;
                if (this.size[1] < MIN_HEIGHT) this.size[1] = MIN_HEIGHT;
                
                return r;
            };
        }
    },
});