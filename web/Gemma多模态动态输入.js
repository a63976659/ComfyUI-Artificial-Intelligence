import { app } from "../../scripts/app.js";

// Gemma 多模态对话节点动态输入: 每类 (图像/视频/音频) 默认只显示 1 个槽位，
// 连接后自动追加下一个空槽位，断开后自动回收多余空槽位 (与 Python 端 INPUT_TYPES 编号一致)。
// 参考 ComfyUI-prompt-formula 的"合并多组提示词"动态输入实现。

const NODE_NAME = "Gemma_Chat";
const GROUPS = [
    { prefix: "图像", type: "IMAGE", max: 9 },
    { prefix: "视频", type: "IMAGE", max: 3 },
    { prefix: "音频", type: "AUDIO", max: 3 },
];

app.registerExtension({
    name: "AI.GemmaChatDynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const groupOf = (name) => GROUPS.find((g) => name.startsWith(g.prefix));

        const updateNodeInputs = function (node) {
            if (!node.inputs) return;
            let hasChanged = false;

            // 1. 收缩: 每类从末尾移除多余的空闲槽位，保留"最后一个已连接 + 1 个空位" (至少 1 个)
            for (const g of GROUPS) {
                const groupIdx = () =>
                    node.inputs.map((inp, i) => (groupOf(inp.name) === g ? i : -1)).filter((i) => i >= 0);
                let lastLink = -1;
                groupIdx().forEach((i, k) => {
                    if (node.inputs[i].link !== null) lastLink = k;
                });
                let target = Math.min(lastLink + 2, g.max);
                if (target < 1) target = 1;

                let remaining = groupIdx().length;
                for (let i = node.inputs.length - 1; i >= 0 && remaining > target; i--) {
                    const inp = node.inputs[i];
                    if (groupOf(inp.name) !== g) continue;
                    if (inp.link !== null) break; // 已连接的槽位一律保留
                    node.removeInput(i);
                    remaining--;
                    hasChanged = true;
                }
            }

            // 2. 增长: 某类最后一个槽位已连接且未到上限时，在末尾追加下一个编号槽位
            for (const g of GROUPS) {
                const idxs = node.inputs
                    .map((inp, i) => (groupOf(inp.name) === g ? i : -1))
                    .filter((i) => i >= 0);
                if (idxs.length === 0) continue;
                const last = node.inputs[idxs[idxs.length - 1]];
                const n = parseInt(last.name.slice(g.prefix.length), 10);
                if (last.link !== null && !isNaN(n) && n < g.max) {
                    node.addInput(`${g.prefix}${n + 1}`, g.type);
                    hasChanged = true;
                }
            }

            // 槽位数量变化后重算节点尺寸并触发重绘
            if (hasChanged) {
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }
        };

        // 节点创建/工作流加载时: 立刻按上述规则整理槽位
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            setTimeout(() => updateNodeInputs(node), 0);
            return r;
        };

        // 连接变化时: 延时等 LiteGraph 内部状态更新完毕再增删槽位
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info, slotObj) {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            if (type === 1) {
                setTimeout(() => updateNodeInputs(this), 20);
            }
            return r;
        };
    },
});
