import server
from aiohttp import web
import tkinter as tk
from tkinter import filedialog
import os

# ================= 1. 导入原有节点 =================
from .节点.translator import LLM_Translator_Node
from .节点.chat import LLM_Chat_Node
from .节点.tts import Qwen_TTS_Node, Qwen_TTS_VoiceDesign_Node, Qwen_TTS_VoiceClone_Node

# ================= 2. 导入新增的音频加载节点 =================
# 确保你的文件结构中 "节点" 文件夹下有 "加载音频.py"
from .节点.加载音频 import 批量加载音频_Node, 加载音频_Node


# ================= 3. API 扩展: 文件/文件夹浏览功能 =================
# 这些接口是前端 JS 按钮 (browse_file/browse_folder) 必须的

@server.PromptServer.instance.routes.post("/qwen/browse_folder")
async def browse_folder(request):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory()
        root.destroy()
        if folder_path:
            return web.json_response({"path": folder_path.replace("\\", "/")})
        return web.json_response({"path": ""})
    except Exception as e:
        return web.json_response({"error": str(e)})

@server.PromptServer.instance.routes.post("/qwen/browse_file")
async def browse_file(request):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        # 限制只显示音频文件
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("All Files", "*.*")]
        )
        root.destroy()
        if file_path:
            return web.json_response({"path": file_path.replace("\\", "/")})
        return web.json_response({"path": ""})
    except Exception as e:
        return web.json_response({"error": str(e)})


# ================= 4. 节点映射定义 =================

NODE_CLASS_MAPPINGS = {
    # --- 文本/翻译类 ---
    "LLM_Translator": LLM_Translator_Node,
    "LLM_Chat": LLM_Chat_Node,
    
    # --- TTS 合成类 ---
    "Qwen_TTS": Qwen_TTS_Node,
    "Qwen_TTS_VoiceDesign": Qwen_TTS_VoiceDesign_Node,
    "Qwen_TTS_VoiceClone": Qwen_TTS_VoiceClone_Node,

    # --- 音频加载类 (新增) ---
    "批量加载音频": 批量加载音频_Node,
    "加载音频": 加载音频_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # --- 文本/翻译类 ---
    "LLM_Translator": "🧠 LLM 智能翻译 (Qwen)",
    "LLM_Chat": "💬 LLM 智能对话 (Qwen)",
    
    # --- TTS 合成类 ---
    "Qwen_TTS": "🔊 Qwen 语音合成 (CustomVoice)",
    "Qwen_TTS_VoiceDesign": "🔊 Qwen 语音设计 (VoiceDesign)",
    "Qwen_TTS_VoiceClone": "🔊 Qwen 语音克隆 (VoiceClone)",

    # --- 音频加载类 (新增) ---
    "批量加载音频": "📂 批量加载音频 (Batch Loader)",
    "加载音频": "🎵 加载音频 (Audio Loader)"
}

# 指定 Web 目录，确保 JS 扩展能被加载
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]