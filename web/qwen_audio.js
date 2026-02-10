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

                // --- 2. 专用功能：音频编辑器 ---
                if (nodeData.name === "加载音频") {
                    
                    let audioBuffer = null;
                    let audioDuration = 0;
                    let dragTarget = null;
                    let isHovering = null;

                    // --- UI 构建 ---
                    const container = document.createElement("div");
                    // 使用 Flex 布局，height: 100% 确保填满节点区域
                    container.style.cssText = "display:flex; flex-direction:column; gap:5px; width:100%; height:100%; min-height:120px; box-sizing:border-box; padding:6px; background:#1a1a1a; border-radius:4px; border:1px solid #333;";
                    
                    const canvas = document.createElement("canvas");
                    // flex-grow: 1 让 Canvas 自动占据剩余高度
                    canvas.style.cssText = "width:100%; flex-grow:1; background:#000; border-radius:3px; display:block; cursor:default;";
                    
                    const audio = document.createElement("audio");
                    audio.controls = true;
                    // 固定播放器高度
                    audio.style.cssText = "width:100%; height:32px; flex-shrink:0; display:block;";

                    container.appendChild(canvas);
                    container.appendChild(audio);
                    
                    // 挂载 Widget
                    this.addDOMWidget("audio_visualizer", "visualizer", container);

                    // 辅助：时间格式化
                    const formatTime = (seconds) => {
                        const m = Math.floor(seconds / 60);
                        const s = Math.floor(seconds % 60);
                        const ms = Math.floor((seconds % 1) * 10);
                        return `${m}:${s.toString().padStart(2, '0')}.${ms}`;
                    };

                    // --- 核心绘制 ---
                    const draw = () => {
                        // 动态获取当前 Canvas 的渲染尺寸
                        const width = canvas.width;
                        const height = canvas.height;
                        const ctx = canvas.getContext("2d");

                        // 1. 清空背景
                        ctx.fillStyle = "#111";
                        ctx.fillRect(0, 0, width, height);

                        if (!audioBuffer || audioDuration === 0) {
                            ctx.fillStyle = "#555";
                            ctx.font = "14px Arial";
                            ctx.textAlign = "center";
                            ctx.fillText("等待音频加载...", width / 2, height / 2);
                            return;
                        }

                        // 获取参数
                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        
                        const startTime = startWidget ? startWidget.value : 0;
                        const duration = durWidget ? durWidget.value : 0;
                        
                        // 计算结束时间 (如果 duration 为 0，则为音频总长)
                        let endTime = (duration > 0.001) ? (startTime + duration) : audioDuration;
                        // 防止越界
                        if (endTime > audioDuration) endTime = audioDuration;

                        // 坐标计算比例
                        const pxPerSec = width / audioDuration;
                        const startX = startTime * pxPerSec;
                        const endX = endTime * pxPerSec;

                        // 2. 绘制波形
                        const raw = audioBuffer.getChannelData(0);
                        const step = Math.ceil(raw.length / width); // 采样步长
                        const amp = (height - 30) / 2; // 振幅高度 (留出上下刻度空间)

                        ctx.beginPath();
                        ctx.strokeStyle = "#4ade80"; 
                        ctx.lineWidth = 1;
                        for (let i = 0; i < width; i++) {
                            let min = 1.0, max = -1.0;
                            for (let j = 0; j < step; j++) {
                                // 简单边界检查防止 undefined
                                const idx = (i * step) + j;
                                if (idx < raw.length) {
                                    const datum = raw[idx];
                                    if (datum < min) min = datum;
                                    if (datum > max) max = datum;
                                }
                            }
                            // 垂直居中绘制
                            ctx.moveTo(i, 15 + amp + (min * amp));
                            ctx.lineTo(i, 15 + amp + (max * amp));
                        }
                        ctx.stroke();

                        // 3. 阴影遮罩 (非选中区域)
                        ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
                        if (startX > 0) ctx.fillRect(0, 0, startX, height);
                        if (endX < width) ctx.fillRect(endX, 0, width - endX, height);

                        // 4. 时间刻度
                        ctx.fillStyle = "#888";
                        ctx.font = "10px monospace";
                        ctx.textAlign = "center";
                        ctx.strokeStyle = "#444";
                        
                        // 根据时长动态调整刻度密度
                        let timeStep = 1;
                        if (audioDuration > 10) timeStep = 2;
                        if (audioDuration > 30) timeStep = 5;
                        if (audioDuration > 120) timeStep = 15;
                        if (audioDuration > 600) timeStep = 60;

                        // 绘制循环
                        for (let t = 0; t <= audioDuration; t += timeStep) {
                            const x = t * pxPerSec;
                            // 上刻度
                            ctx.beginPath();
                            ctx.moveTo(x, 0); ctx.lineTo(x, 8);
                            ctx.stroke();
                            // 下刻度
                            ctx.beginPath();
                            ctx.moveTo(x, height); ctx.lineTo(x, height - 8);
                            ctx.stroke();
                            
                            // 文字 (避免过密)
                            if (x < width - 10) { 
                                ctx.fillText(formatTime(t), x, 20);
                            }
                        }

                        // 5. 绘制滑块 (Start - Cyan)
                        ctx.lineWidth = 2;
                        ctx.strokeStyle = (dragTarget === 'start' || isHovering === 'start') ? "#ffffff" : "#00ffff";
                        ctx.fillStyle = ctx.strokeStyle;
                        
                        ctx.beginPath();
                        ctx.moveTo(startX, 0); ctx.lineTo(startX, height);
                        ctx.stroke();
                        // 顶部手柄
                        ctx.beginPath();
                        ctx.moveTo(startX, 0); ctx.lineTo(startX+8, 0); ctx.lineTo(startX, 10); ctx.fill();

                        // 6. 绘制滑块 (End - Red)
                        ctx.strokeStyle = (dragTarget === 'end' || isHovering === 'end') ? "#ffffff" : "#ff0055";
                        ctx.fillStyle = ctx.strokeStyle;

                        ctx.beginPath();
                        ctx.moveTo(endX, 0); ctx.lineTo(endX, height);
                        ctx.stroke();
                        // 底部手柄
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
                        // 播放范围约束
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
                    
                    // 坐标转换 (处理缩放)
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
                        
                        const threshold = 20; // 判定范围

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
                        
                        // 悬停检测
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

                    // --- 关键修复1：监听容器大小变化 (响应式拉伸) ---
                    // 使用 ResizeObserver 替代旧的 onResize，更精准
                    const resizeObserver = new ResizeObserver(entries => {
                        for (let entry of entries) {
                            // 获取容器当前的内容区域大小
                            const { width, height } = entry.contentRect;
                            
                            // 减去 Audio 播放器的高度(32px) 和一些 padding，赋予 Canvas
                            // 确保 Canvas 像素分辨率与显示尺寸匹配
                            if (width > 0 && height > 0) {
                                canvas.width = width;
                                // 至少给 Canvas 留 50px，避免报错
                                canvas.height = Math.max(50, height - 35); 
                                requestAnimationFrame(draw);
                            }
                        }
                    });
                    // 开始监听容器
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

                        // --- 关键修复2：切换音频时重置状态 ---
                        // 1. 重置 Widget 值为 0 (全选)，防止红线错位
                        const startWidget = this.widgets.find(w => w.name === "开始时间");
                        const durWidget = this.widgets.find(w => w.name === "持续时间");
                        
                        if (startWidget) { startWidget.value = 0; startWidget.callback(0); }
                        if (durWidget) { durWidget.value = 0; durWidget.callback(0); }

                        // 2. 清空当前波形缓存，避免绘制错误的旧图
                        audioBuffer = null;
                        audioDuration = 0;
                        requestAnimationFrame(draw); 

                        // 3. 加载新文件
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

                    // Queue 执行后
                    const onExecuted = nodeType.prototype.onExecuted;
                    nodeType.prototype.onExecuted = function (message) {
                        onExecuted?.apply(this, arguments);
                        if (message?.audio?.[0]) {
                            const info = message.audio[0];
                            const url = api.api_base + `/view?filename=${encodeURIComponent(info.filename)}&type=${info.type}&subfolder=${info.subfolder}`;
                            audio.src = url;
                            // 此时不重置参数，因为用户可能在听裁剪后的效果
                        }
                    };

                    // 初始化一次默认大小，之后由 ResizeObserver 接管
                    setTimeout(() => { this.setSize([400, 320]); }, 50);
                }
                return r;
            };
        }
    },
});