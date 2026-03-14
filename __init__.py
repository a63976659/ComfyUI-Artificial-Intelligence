# ================= 1. 激活所有后端 API 路由 =================
from . import server_api

# ================= 2. 导入所有节点类 =================
from .节点.translator import LLM_Translator_Node
from .节点.chat import LLM_Chat_Node
from .节点.tts import Qwen_TTS_Node, Qwen_TTS_VoiceDesign_Node, Qwen_TTS_VoiceClone_Node
from .节点.加载音频 import 批量加载音频_Node, 加载音频_Node
from .节点.加载视频 import 加载视频_Node, 裁剪视频_Node

# ================= 3. 节点映射定义 =================
NODE_CLASS_MAPPINGS = {
    "LLM_Translator": LLM_Translator_Node,
    "LLM_Chat": LLM_Chat_Node,
    "Qwen_TTS": Qwen_TTS_Node,
    "Qwen_TTS_VoiceDesign": Qwen_TTS_VoiceDesign_Node,
    "Qwen_TTS_VoiceClone": Qwen_TTS_VoiceClone_Node,
    "批量加载音频": 批量加载音频_Node,
    "加载音频": 加载音频_Node,
    "加载视频": 加载视频_Node,
    "裁剪视频": 裁剪视频_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_Translator": "🧠 LLM 智能翻译 (Qwen)",
    "LLM_Chat": "💬 LLM 智能对话 (Qwen)",
    "Qwen_TTS": "🔊 Qwen 语音合成 (CustomVoice)",
    "Qwen_TTS_VoiceDesign": "🔊 Qwen 语音设计 (VoiceDesign)",
    "Qwen_TTS_VoiceClone": "🔊 Qwen 语音克隆 (VoiceClone)",
    "批量加载音频": "📂 批量加载音频 (Batch Loader)",
    "加载音频": "🎵 加载音频 (Audio Loader)",
    "加载视频": "🎬 加载视频 (Load Video)",
    "裁剪视频": "✂️ 裁剪视频 (Crop Video)"
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]