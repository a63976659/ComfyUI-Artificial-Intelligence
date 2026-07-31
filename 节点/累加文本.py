# -*- coding: utf-8 -*-
"""
持续填充文本节点
多行文本框直接显示在节点上，作为译文/文本的持续汇总区，两种填充方式:
- 图执行模式: 连接上游文本 (如实时翻译的"翻译结果")，每次运行把新文本追加一行
- 真·实时会话: 实时翻译会话每完成一句，译文直接追加一行 (不经过图执行)
两者共用 ai_text_accumulate 事件同步前端文本框内容。
"""
from server import PromptServer

ACCUMULATE_EVENT = "ai_text_accumulate"


def push_append(node_id, text):
    """向指定节点的文本框追加一行 (供实时会话调用，不经过图执行)"""
    if not node_id or not text:
        return
    PromptServer.instance.send_sync(
        ACCUMULATE_EVENT, {"node": str(node_id), "text": text, "append": True}
    )


class 累加文本_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本内容": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "文本会持续追加到这里 (可手动编辑/清空)",
                }),
            },
            "optional": {
                "追加文本": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "fill"
    CATEGORY = "💬 AI人工智能/实时翻译"
    DESCRIPTION = (
        "持续填充文本节点：多行文本框直接显示在节点上，新文本每次自动换行追加。"
        "把 🌐 实时翻译 的\"翻译结果\"连到\"追加文本\"输入即可持续累加译文——"
        "真·实时会话模式下每翻译完一句自动追加，无需点击运行。"
        "文本框可手动编辑或清空。"
    )

    def fill(self, 文本内容, 追加文本=None, unique_id=None):
        新增 = (追加文本 or "").strip()
        文本 = 文本内容
        if 新增:
            文本 = f"{文本}\n{新增}" if 文本.strip() else 新增
            # 回写前端文本框，使累加结果持久可见 (append=False 表示整体覆写)
            if unique_id is not None:
                PromptServer.instance.send_sync(
                    ACCUMULATE_EVENT,
                    {"node": str(unique_id), "text": 文本, "append": False},
                )
        return (文本,)
