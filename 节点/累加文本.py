# -*- coding: utf-8 -*-
"""
持续填充文本节点
多行文本框直接显示在节点上，作为译文/文本的持续汇总区，两种填充方式:
- 图执行模式: 连接上游文本 (如实时翻译的"翻译结果"/"识别结果")，每次运行把新文本追加一块
- 真·实时会话: 实时翻译会话每完成一句，原文 + 译文直接成对追加 (不经过图执行)
同时接入"识别文本"与"追加文本"时按 🗣原文 / ➜译文 双语成对存储，与实时翻译显示区一致。
两者共用 ai_text_accumulate 事件同步前端文本框内容。
"""
from server import PromptServer

ACCUMULATE_EVENT = "ai_text_accumulate"


def _make_block(source, translation):
    """把原文/译文组成一块文本: 两者都有→双语成对，否则取非空的一侧"""
    src = (source or "").strip()
    tr = (translation or "").strip()
    if src and tr:
        return f"🗣 {src}\n➜ {tr}"
    return tr or src


def push_append(node_id, source, translation=None):
    """向指定节点的文本框追加一块 (供实时会话调用，不经过图执行)

    传两个参数时按 🗣原文 / ➜译文 双语成对追加；只传单个则直接追加该文本。
    """
    if not node_id:
        return
    block = _make_block(source, translation)
    if not block:
        return
    PromptServer.instance.send_sync(
        ACCUMULATE_EVENT, {"node": str(node_id), "text": block, "append": True}
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
                "识别文本": ("STRING", {"forceInput": True}),
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
        "把 🌐 实时翻译 的“翻译结果”连到“追加文本”即可持续累加译文；"
        "同时把“识别结果”连到“识别文本”则按 🗣原文 / ➜译文 双语成对记录——"
        "真·实时会话模式下每翻译完一句自动追加，无需点击运行。文本框可手动编辑或清空。"
    )

    def fill(self, 文本内容, 追加文本=None, 识别文本=None, unique_id=None):
        新块 = _make_block(识别文本, 追加文本)  # 双语成对 / 单侧
        文本 = 文本内容
        if 新块:
            文本 = f"{文本}\n\n{新块}" if 文本.strip() else 新块
            # 回写前端文本框，使累加结果持久可见 (append=False 表示整体覆写)
            if unique_id is not None:
                PromptServer.instance.send_sync(
                    ACCUMULATE_EVENT,
                    {"node": str(unique_id), "text": 文本, "append": False},
                )
        return (文本,)
