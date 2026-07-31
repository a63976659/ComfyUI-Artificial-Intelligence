import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 实时翻译节点前端扩展: 监听后端流式事件，在节点显示区打字机式刷新译文

const NODE_NAME = "LLM_Realtime_Translator";
const STREAM_EVENT = "ai_realtime_translate";

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

            return r;
        };
    },

    setup() {
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
