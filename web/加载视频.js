import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Qwen.VideoLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        const nodeConfig = {
            "加载视频": { widgetName: "文件路径", apiRoute: "/qwen/browse_file", btnText: "🎬 浏览视频" },
            "裁剪视频": { widgetName: "文件路径", apiRoute: "/qwen/browse_file", btnText: "✂️ 浏览视频" }
        };

        if (nodeConfig[nodeData.name]) {
            const config = nodeConfig[nodeData.name];
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const MIN_WIDTH = 400;
                const MIN_HEIGHT = 400; 

                const pathWidget = this.widgets.find((w) => w.name === config.widgetName);
                if (pathWidget) {
                    if (pathWidget.inputEl) {
                        pathWidget.inputEl.style.display = "none";
                        pathWidget.inputEl.style.opacity = "0";
                    }
                    pathWidget.computeSize = () => [0, -4];
                    pathWidget.draw = () => {};

                    this.addWidget("button", config.btnText, null, () => {
                        api.fetchApi(config.apiRoute, { method: "POST" })
                        .then(r => r.json())
                        .then(data => { if (data.path) { pathWidget.value = data.path; pathWidget.callback(data.path); }})
                        .catch(e => console.error(e));
                    });
                }

                // ==========================================
                // UI 构建
                // ==========================================
                let videoDuration = 0;
                let dragTarget = null;
                let isHovering = null;
                let lastLoadedPath = "";
                let _isConfiguring = false; // 状态锁：判断是否正在恢复工作流

                const container = document.createElement("div");
                container.style.cssText = "display:flex; flex-direction:column; gap:6px; width:100%; height:100%; box-sizing:border-box; padding:8px; background:#1a1a1a; border-radius:4px; border:1px solid #333;";
                
                const titleBar = document.createElement("div");
                titleBar.style.cssText = "background:#222; color:#eee; text-align:center; padding:6px 10px; border-radius:4px; font-size:13px; font-family:sans-serif; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex-shrink:0; border:1px solid #111;";
                titleBar.innerText = "未选择视频";

                const video = document.createElement("video");
                video.controls = false; 
                video.playsInline = true; 
                video.style.cssText = "width:100%; flex-grow:1; min-height:0; object-fit:contain; background:#000; border-radius:4px; display:block; cursor:pointer;";
                video.onclick = () => { video.paused ? video.play() : video.pause(); };
                
                const controlBar = document.createElement("div");
                controlBar.style.cssText = "display:flex; gap:6px; height: 30px; width: 100%; align-items:center; flex-shrink:0;";
                
                const playBtn = document.createElement("button");
                playBtn.innerText = "▶";
                playBtn.style.cssText = "background:#444; color:#fff; border:none; border-radius:3px; cursor:pointer; width:30px; height:100%; flex-shrink:0;";
                playBtn.onclick = (e) => { 
                    e.stopPropagation(); 
                    video.paused ? video.play() : video.pause(); 
                };

                const canvas = document.createElement("canvas");
                canvas.style.cssText = "height:100%; background:#000; border-radius:3px; display:block; cursor:default; flex-grow:1; min-width:0;";

                const fpsLabel = document.createElement("div");
                fpsLabel.style.cssText = "color:#888; font-size:11px; text-align:right; font-family:monospace; padding-right:4px; flex-shrink:0; min-width:50px;";
                fpsLabel.innerText = "FPS: --";

                controlBar.appendChild(playBtn);
                controlBar.appendChild(canvas);
                controlBar.appendChild(fpsLabel);
                
                container.appendChild(titleBar);
                container.appendChild(video);
                container.appendChild(controlBar);
                
                this.addDOMWidget("video_visualizer", "visualizer", container);

                const formatTime = (sec) => {
                    const m = Math.floor(sec / 60);
                    const s = Math.floor(sec % 60);
                    const ms = Math.floor((sec % 1) * 10);
                    return `${m}:${s.toString().padStart(2, '0')}.${ms}`;
                };

                const draw = () => {
                    const width = canvas.width;
                    const height = canvas.height;
                    const ctx = canvas.getContext("2d");

                    ctx.fillStyle = "#111";
                    ctx.fillRect(0, 0, width, height);

                    if (videoDuration === 0) {
                        ctx.fillStyle = "#555";
                        ctx.font = "12px Arial";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText("等待视频...", width / 2, height / 2);
                        return;
                    }

                    const pxPerSec = width / videoDuration;

                    ctx.fillStyle = "#888";
                    ctx.font = "9px monospace";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "top";
                    ctx.strokeStyle = "#444";
                    
                    let timeStep = videoDuration > 60 ? 10 : (videoDuration > 10 ? 2 : 1);
                    for (let t = 0; t <= videoDuration; t += timeStep) {
                        const x = t * pxPerSec;
                        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 5); ctx.stroke();
                        if (x < width - 10) ctx.fillText(formatTime(t), x, 7);
                    }

                    const playX = video.currentTime * pxPerSec;
                    ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
                    ctx.fillRect(0, 0, playX, height);
                    ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
                    ctx.beginPath(); ctx.moveTo(playX, 0); ctx.lineTo(playX, height); ctx.stroke();

                    if (nodeData.name === "加载视频") {
                        const currWidget = this.widgets.find(w => w.name === "当前时间");
                        const currTime = currWidget ? parseFloat(currWidget.value) : 0;
                        const currX = currTime * pxPerSec;

                        ctx.strokeStyle = (dragTarget === 'curr' || isHovering === 'curr') ? "#fff" : "#00ffff";
                        ctx.lineWidth = 2;
                        ctx.beginPath(); ctx.moveTo(currX, 0); ctx.lineTo(currX, height); ctx.stroke();
                        ctx.fillStyle = ctx.strokeStyle;
                        ctx.beginPath(); ctx.moveTo(currX, 0); ctx.lineTo(currX+6, 0); ctx.lineTo(currX, 6); ctx.fill();

                    } else if (nodeData.name === "裁剪视频") {
                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        const sTime = startWidget ? parseFloat(startWidget.value) : 0;
                        const dTime = durWidget ? parseFloat(durWidget.value) : 0;
                        let eTime = (dTime > 0.001) ? sTime + dTime : videoDuration;
                        if (eTime > videoDuration) eTime = videoDuration;

                        const sX = sTime * pxPerSec;
                        const eX = eTime * pxPerSec;

                        ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
                        ctx.fillRect(0, 0, sX, height);
                        ctx.fillRect(eX, 0, width - eX, height);

                        ctx.strokeStyle = (dragTarget === 'start' || isHovering === 'start') ? "#fff" : "#00ffff";
                        ctx.lineWidth = 2;
                        ctx.beginPath(); ctx.moveTo(sX, 0); ctx.lineTo(sX, height); ctx.stroke();
                        ctx.fillStyle = ctx.strokeStyle;
                        ctx.beginPath(); ctx.moveTo(sX, 0); ctx.lineTo(sX+6, 0); ctx.lineTo(sX, 6); ctx.fill();

                        ctx.strokeStyle = (dragTarget === 'end' || isHovering === 'end') ? "#fff" : "#ff0055";
                        ctx.beginPath(); ctx.moveTo(eX, 0); ctx.lineTo(eX, height); ctx.stroke();
                        ctx.fillStyle = ctx.strokeStyle;
                        ctx.beginPath(); ctx.moveTo(eX, height); ctx.lineTo(eX-6, height); ctx.lineTo(eX, height-6); ctx.fill();
                    }
                };
                
                video.addEventListener('play', () => { playBtn.innerText = "⏸"; });
                video.addEventListener('pause', () => { playBtn.innerText = "▶"; });

                video.addEventListener('timeupdate', () => {
                    if (nodeData.name === "裁剪视频" && !video.paused) {
                        const sW = this.widgets.find(w => w.name === "开始时间");
                        const dW = this.widgets.find(w => w.name === "持续时间");
                        if (sW && dW && parseFloat(dW.value) > 0.001) {
                            if (video.currentTime >= parseFloat(sW.value) + parseFloat(dW.value)) {
                                video.pause();
                                video.currentTime = parseFloat(sW.value);
                            }
                        }
                    }
                    requestAnimationFrame(draw);
                });

                const getCanvasX = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    return (e.clientX - rect.left) * (canvas.width / rect.width);
                };

                canvas.addEventListener("mousedown", (e) => {
                    if (!videoDuration) return;
                    const x = getCanvasX(e);
                    const width = canvas.width;
                    const pxPerSec = width / videoDuration;
                    const threshold = 20;

                    if (nodeData.name === "加载视频") {
                        dragTarget = 'curr';
                        updateWidgets(x, dragTarget);
                        const cW = this.widgets.find(w => w.name === "当前时间");
                        video.currentTime = cW ? (parseFloat(cW.value) || 0) : 0;
                    } else if (nodeData.name === "裁剪视频") {
                        const sW = this.widgets.find(w => w.name === "开始时间");
                        const dW = this.widgets.find(w => w.name === "持续时间");
                        
                        const sVal = parseFloat(sW.value) || 0;
                        const dVal = parseFloat(dW.value) || 0;
                        const sX = sVal * pxPerSec;
                        const eX = (dVal > 0.001 ? sVal + dVal : videoDuration) * pxPerSec;

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
                            video.currentTime = (parseFloat(sW.value) || 0) + (parseFloat(dW.value) || 0);
                        } else {
                            video.currentTime = parseFloat(sW.value) || 0;
                        }
                    }
                    requestAnimationFrame(draw);
                });

                window.addEventListener("mousemove", (e) => {
                    if (!videoDuration) return;
                    if (dragTarget) {
                        const x = getCanvasX(e);
                        updateWidgets(x, dragTarget);
                        
                        if (nodeData.name === "加载视频") {
                            const cW = this.widgets.find(w => w.name === "当前时间");
                            video.currentTime = cW ? (parseFloat(cW.value) || 0) : 0;
                        } else if (nodeData.name === "裁剪视频") {
                            const sW = this.widgets.find(w => w.name === "开始时间");
                            const dW = this.widgets.find(w => w.name === "持续时间");
                            const sVal = parseFloat(sW.value) || 0;
                            const dVal = parseFloat(dW.value) || 0;
                            
                            if (dragTarget === 'end') {
                                video.currentTime = sVal + dVal; 
                            } else {
                                video.currentTime = sVal; 
                            }
                        }
                    } 
                    
                    const rect = canvas.getBoundingClientRect();
                    const mx = e.clientX - rect.left;
                    if (mx >= 0 && mx <= rect.width && e.clientY - rect.top >= 0 && e.clientY - rect.top <= rect.height) {
                        const x = getCanvasX(e);
                        const pxPerSec = canvas.width / videoDuration;
                        let nCursor = "crosshair", nHover = null;

                        if (nodeData.name === "加载视频") {
                            const cW = this.widgets.find(w => w.name === "当前时间");
                            const cX = (parseFloat(cW.value) || 0) * pxPerSec;
                            if (Math.abs(x - cX) < 20) { nCursor = "ew-resize"; nHover = 'curr'; }
                        } else {
                            const sW = this.widgets.find(w => w.name === "开始时间");
                            const dW = this.widgets.find(w => w.name === "持续时间");
                            const sVal = parseFloat(sW.value) || 0;
                            const dVal = parseFloat(dW.value) || 0;
                            
                            const sX = sVal * pxPerSec;
                            const eX = (dVal > 0.001 ? sVal + dVal : videoDuration) * pxPerSec;
                            
                            if (Math.abs(x - eX) < 20) { nCursor = "ew-resize"; nHover = 'end'; }
                            else if (Math.abs(x - sX) < 20) { nCursor = "ew-resize"; nHover = 'start'; }
                        }
                        canvas.style.cursor = nCursor;
                        if (isHovering !== nHover) { isHovering = nHover; requestAnimationFrame(draw); }
                    } else {
                        isHovering = null; requestAnimationFrame(draw);
                    }
                });

                window.addEventListener("mouseup", () => { 
                    if (dragTarget) {
                        if (nodeData.name === "裁剪视频") {
                            const sW = this.widgets.find(w => w.name === "开始时间");
                            if (sW) video.currentTime = parseFloat(sW.value) || 0;
                        }
                    }
                    dragTarget = null; 
                    requestAnimationFrame(draw); 
                });

                const updateWidgets = (mouseX, mode) => {
                    const width = canvas.width;
                    let safeX = Math.max(0, Math.min(mouseX, width));
                    const time = (safeX / width) * videoDuration;
                    
                    if (nodeData.name === "加载视频" && mode === 'curr') {
                        const cW = this.widgets.find(w => w.name === "当前时间");
                        cW.value = parseFloat(time.toFixed(2));
                        cW.callback(cW.value);
                    } else if (nodeData.name === "裁剪视频") {
                        const sW = this.widgets.find(w => w.name === "开始时间");
                        const dW = this.widgets.find(w => w.name === "持续时间");
                        
                        let s = parseFloat(sW.value) || 0;
                        let d = parseFloat(dW.value) || 0;
                        let e_time = (d > 0.001) ? (s + d) : videoDuration;

                        if (mode === 'start' || mode === 'start_jump') {
                            let nS = parseFloat(time.toFixed(2));
                            if (nS >= e_time) nS = Math.max(0, e_time - 0.05);
                            
                            sW.value = parseFloat(nS.toFixed(2)); 
                            sW.callback(sW.value);
                            
                            let nD = parseFloat((e_time - sW.value).toFixed(2));
                            dW.value = Math.max(0, nD); 
                            dW.callback(dW.value);

                        } else if (mode === 'end') {
                            let nE = parseFloat(time.toFixed(2));
                            if (nE <= s) nE = s + 0.05;
                            
                            let nD = parseFloat((Math.min(nE, videoDuration) - s).toFixed(2));
                            dW.value = Math.max(0, nD); 
                            dW.callback(dW.value);
                        }
                    }
                    requestAnimationFrame(draw);
                };

                const resizeObserver = new ResizeObserver(entries => {
                    for (let entry of entries) {
                        const { width } = entry.contentRect;
                        if (width > 0 && width !== canvas.width) {
                            canvas.width = width;
                            canvas.height = 30; 
                            requestAnimationFrame(draw);
                        }
                    }
                });
                resizeObserver.observe(canvas);

                const getMediaUrl = (inputPath) => {
                    if (!inputPath) return "";
                    return api.api_base + `/qwen/view_media?path=${encodeURIComponent(inputPath)}`;
                };

                const updatePreview = (filePath, isInit = false) => {
                    if (!filePath) {
                        titleBar.innerText = "未选择视频";
                        return;
                    }
                    
                    titleBar.innerText = filePath.replace(/\\/g, '/').split('/').pop();

                    // 【核心修复1】防止加载相同视频导致进度清零
                    if (filePath === lastLoadedPath) return; 
                    lastLoadedPath = filePath;

                    // 【核心修复2】如果当前正在读取工作流配置，绝对不重置参数！
                    const isGraphLoading = app.configuringGraph || _isConfiguring;
                    if (!isInit && !isGraphLoading) {
                        if (nodeData.name === "加载视频") {
                            const cW = this.widgets.find(w => w.name === "当前时间");
                            if (cW) { cW.value = 0; cW.callback(0); }
                        } else {
                            const sW = this.widgets.find(w => w.name === "开始时间");
                            const dW = this.widgets.find(w => w.name === "持续时间");
                            if (sW) { sW.value = 0; sW.callback(0); }
                            if (dW) { dW.value = 0; dW.callback(0); }
                        }
                    }

                    videoDuration = 0;
                    fpsLabel.innerText = "FPS: --"; 
                    video.src = getMediaUrl(filePath) + `&t=${Date.now()}`;
                    video.load();

                    api.fetchApi(`/qwen/video_metadata?path=${encodeURIComponent(filePath)}`)
                    .then(r => r.json())
                    .then(data => {
                        if (data && data.fps) {
                            fpsLabel.innerText = `FPS: ${data.fps.toFixed(2)}`;
                        }
                    })
                    .catch(e => console.error("获取FPS失败", e));
                };

                video.onloadedmetadata = () => {
                    videoDuration = video.duration;
                    
                    // 【核心修复3】移除强制覆盖 `持续时间` 的逻辑，保留 0 = 全长 的设定
                    
                    if (video.videoWidth > 0 && video.videoHeight > 0) {
                        const aspect = video.videoWidth / video.videoHeight;
                        let targetWidth = this.size[0]; 
                        if (targetWidth < MIN_WIDTH) targetWidth = MIN_WIDTH;
                        
                        const videoH = (targetWidth - 16) / aspect; 
                        const baseHeight = this.computeSize([targetWidth, 0])[1];
                        
                        let totalHeight = baseHeight + videoH + 95; 
                        if (totalHeight < MIN_HEIGHT) totalHeight = MIN_HEIGHT; 
                        
                        // 【核心修复4】如果在恢复工作流中，保护并保留用户自行设定的历史尺寸
                        const isGraphLoading = app.configuringGraph || _isConfiguring;
                        if (!isGraphLoading) {
                            this.setSize([targetWidth, totalHeight]);
                        }
                    }

                    // 加载完毕后，画面精准跳转回之前保存的时间线
                    if (nodeData.name === "加载视频") {
                        const cW = this.widgets.find(w => w.name === "当前时间");
                        if (cW && parseFloat(cW.value) > 0) video.currentTime = parseFloat(cW.value);
                    } else if (nodeData.name === "裁剪视频") {
                        const sW = this.widgets.find(w => w.name === "开始时间");
                        if (sW && parseFloat(sW.value) > 0) video.currentTime = parseFloat(sW.value);
                    }

                    requestAnimationFrame(draw);
                };

                video.onerror = () => {
                    console.error("Video load error! Format might not be supported.", video.error);
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
                    _isConfiguring = true; // 锁定状态，防止在此期间重置参数
                    if (origOnConfigure) origOnConfigure.apply(this, arguments);
                    if (pathWidget && pathWidget.value) {
                        updatePreview(pathWidget.value, true);
                    }
                    _isConfiguring = false; // 解除锁定
                };

                setTimeout(() => { 
                    if (pathWidget && pathWidget.value && videoDuration === 0) {
                        updatePreview(pathWidget.value, true); 
                    }
                }, 500);

                this.widgets.forEach(w => {
                    if (w.name === "当前时间" || w.name === "开始时间" || w.name === "持续时间") {
                        const cb = w.callback;
                        w.callback = function(v) {
                            if (cb) cb.call(this, v);
                            requestAnimationFrame(draw);
                        };
                    }
                });

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

                // 【核心修复5】不再强制覆盖历史保存的尺寸
                if (this.size[0] < MIN_WIDTH) this.size[0] = MIN_WIDTH;
                if (this.size[1] < MIN_HEIGHT) this.size[1] = MIN_HEIGHT;

                return r;
            };
        }
    },
});