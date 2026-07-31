import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 实时翻译节点前端扩展: 监听后端流式事件，在节点显示区打字机式刷新译文

const NODE_NAME = "LLM_Realtime_Translator";
const STREAM_EVENT = "ai_realtime_translate";

// 目标语言随模型联动: 从后端拉取"模型 -> 支持语言"映射 (与后端 TARGET_LANGUAGES 同一数据源)
let LANG_BY_MODEL = null;
let LANG_MAP_PROMISE = null;
function loadLangMap() {
    if (LANG_BY_MODEL) return Promise.resolve(LANG_BY_MODEL);
    if (!LANG_MAP_PROMISE) {
        LANG_MAP_PROMISE = api.fetchApi("/qwen/realtime/languages")
            .then((r) => r.json())
            .then((m) => { LANG_BY_MODEL = m; return m; })
            .catch((err) => { console.warn("[实时翻译] 语言列表拉取失败", err); LANG_MAP_PROMISE = null; return null; });
    }
    return LANG_MAP_PROMISE;
}

app.registerExtension({
    name: "AI.RealtimeTranslator",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // 只读流式显示区
            const el = document.createElement("div");
            el.style.cssText = [
                "width: 100%",
                "height: 100%",
                "box-sizing: border-box",
                "overflow-y: auto",
                "padding: 8px",
                "background: #1a1a2e",
                "color: #7fdb8f",
                "border: 1px solid #444",
                "border-radius: 6px",
                "font-family: Consolas, monospace",
                "font-size: 13px",
                "line-height: 1.5",
                "white-space: pre-wrap",
                "word-break: break-word",
            ].join(";");
            el.textContent = "等待翻译...";
            this._实时翻译显示区 = el;

            this.addDOMWidget("实时显示", "div", el, { serialize: false });

            const minWidth = 360;
            const minHeight = 240;
            if (this.size[0] < minWidth) this.size[0] = minWidth;
            if (this.size[1] < minHeight) this.size[1] = minHeight;

            // —— 目标语言随模型自动收窄可选范围 ——
            const node = this;
            node._syncLangByModel = function () {
                if (!LANG_BY_MODEL || !node.widgets) return;
                const modelW = node.widgets.find((w) => w.name === "模型名称");
                const langW = node.widgets.find((w) => w.name === "目标语言");
                if (!modelW || !langW) return;
                const allowed = LANG_BY_MODEL[modelW.value];
                if (!allowed || !allowed.length) return;
                langW.options.values = allowed;
                if (!allowed.includes(langW.value)) {
                    const fallback = allowed.includes("英文") ? "英文" : allowed[0];
                    langW.value = fallback;
                    if (langW.callback) langW.callback(fallback);
                }
            };
            const modelW = this.widgets.find((w) => w.name === "模型名称");
            if (modelW) {
                const prevCb = modelW.callback;
                modelW.callback = function () {
                    const ret = prevCb ? prevCb.apply(this, arguments) : undefined;
                    node._syncLangByModel();
                    return ret;
                };
            }
            loadLangMap().then(() => node._syncLangByModel());

            return r;
        };
    },

    setup() {
        // 预取语言映射，并对已存在的实时翻译节点应用一次联动收窄
        loadLangMap().then(() => {
            for (const node of app.graph._nodes || []) {
                if (node.comfyClass === NODE_NAME && node._syncLangByModel) node._syncLangByModel();
            }
        });

        // 流式增量: 按节点 id 定位并刷新显示区
        api.addEventListener(STREAM_EVENT, (event) => {
            const detail = event.detail;
            if (!detail || detail.node === undefined) return;
            const node = app.graph.getNodeById(Number(detail.node));
            if (!node || !node._实时翻译显示区) return;
            node._实时翻译显示区.textContent = detail.text || "";
            node._实时翻译显示区.scrollTop = node._实时翻译显示区.scrollHeight;
        });

        // 仅当本节点自身开始执行时才显示"翻译中"，其它节点运行不影响显示区
        api.addEventListener("executing", (event) => {
            const detail = event.detail;
            const nodeId = detail && typeof detail === "object" ? (detail.display_node ?? detail.node) : detail;
            if (nodeId === null || nodeId === undefined) return;
            const node = app.graph.getNodeById(Number(nodeId));
            if (node && node.comfyClass === NODE_NAME && node._实时翻译显示区) {
                node._实时翻译显示区.textContent = "翻译中...";
            }
        });
    },
});
