import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 持续填充文本节点前端扩展: 监听累加事件，把新文本换行追加到节点的多行文本框

const NODE_NAME = "Text_Accumulator";
const ACCUMULATE_EVENT = "ai_text_accumulate";
const TEXT_WIDGET = "文本内容";

app.registerExtension({
    name: "AI.TextAccumulator",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            // 默认尺寸放大，便于直接在节点上阅读累加的多行文本
            const minWidth = 360;
            const minHeight = 260;
            if (this.size[0] < minWidth) this.size[0] = minWidth;
            if (this.size[1] < minHeight) this.size[1] = minHeight;
            return r;
        };
    },

    setup() {
        api.addEventListener(ACCUMULATE_EVENT, (event) => {
            const detail = event.detail;
            if (!detail || detail.node === undefined) return;
            const node = app.graph.getNodeById(Number(detail.node));
            if (!node || node.comfyClass !== NODE_NAME) return;
            const widget = node.widgets && node.widgets.find((w) => w.name === TEXT_WIDGET);
            if (!widget) return;

            const incoming = detail.text || "";
            if (detail.append) {
                // 实时会话: 每句译文追加一行 (首行不留空行)
                const current = widget.value || "";
                widget.value = current.trim() ? `${current}\n${incoming}` : incoming;
            } else {
                // 图执行回写: 整体覆写为后端累加后的完整文本
                widget.value = incoming;
            }

            // 滚动到底部，始终可见最新一行
            const el = widget.inputEl || (widget.element && widget.element.querySelector("textarea"));
            if (el) el.scrollTop = el.scrollHeight;
            node.setDirtyCanvas(true, false);
        });
    },
});
