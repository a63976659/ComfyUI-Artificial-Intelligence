# -*- coding: utf-8 -*-
"""
麦克风录音节点
配合 web/录音.js 使用，提供两种模式:
1. 一次性录音: 前端采集麦克风 -> 编码 WAV 上传临时目录 (重启自动清理，不永久占用 input) ->
   本节点在图执行时读取并输出 AUDIO，可连接实时翻译节点，或连接 ComfyUI 原生"保存音频"节点永久保存。
2. 真·实时翻译: 点击节点上的"🌐 开始实时翻译"按钮，前端持续采集 + VAD 自动断句，
   每句直接上传后端实时会话 (见 实时翻译.py 的 /qwen/realtime/* 路由)，
   不经过图执行，译文持续显示在下游实时翻译节点上。
"""
import os

import torch
import soundfile as sf
import folder_paths


class 录音_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "录音文件": ("STRING", {"default": "", "placeholder": "点击下方按钮录音，文件名自动填入"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "load_record"
    CATEGORY = "💬 AI人工智能/实时翻译"
    DESCRIPTION = (
        "麦克风录音节点：点击\"🎙️ 开始录音\"可一次性录音 (需浏览器授权)，输出 AUDIO。"
        "录音仅存于临时目录 (重启自动清理)，如需永久保存请把\"音频\"输出连到 ComfyUI 原生\"保存音频\"节点。"
        "将本节点音频输出连到实时翻译节点后，点击\"🌐 开始实时翻译\"即进入真·实时模式："
        "无需点击运行，边说边自动断句，译文持续显示在实时翻译节点上，再次点击停止。"
    )

    def load_record(self, 录音文件):
        if not 录音文件.strip():
            raise ValueError("尚未录音，请先点击节点上的\"开始录音\"按钮")

        file_path = os.path.join(folder_paths.get_temp_directory(), 录音文件)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"录音文件不存在: {file_path}，请重新录音")

        # soundfile 读出 (frames, channels)，转为 ComfyUI 的 (1, channels, frames)
        data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T).unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": int(sample_rate)},)

    @classmethod
    def IS_CHANGED(cls, 录音文件):
        """以文件修改时间为指纹，重新录音后节点自动重新执行"""
        if not 录音文件.strip():
            return ""
        file_path = os.path.join(folder_paths.get_temp_directory(), 录音文件)
        if os.path.isfile(file_path):
            return str(os.path.getmtime(file_path))
        return ""
