from .节点.translator import LLM_Translator_Node
from .节点.chat import LLM_Chat_Node
# 引入所有三个 TTS 节点类
from .节点.tts import Qwen_TTS_Node, Qwen_TTS_VoiceDesign_Node, Qwen_TTS_VoiceClone_Node

NODE_CLASS_MAPPINGS = {
    # 文本/翻译类
    "LLM_Translator": LLM_Translator_Node,
    "LLM_Chat": LLM_Chat_Node,
    
    # 音频类
    "Qwen_TTS": Qwen_TTS_Node,
    "Qwen_TTS_VoiceDesign": Qwen_TTS_VoiceDesign_Node,
    "Qwen_TTS_VoiceClone": Qwen_TTS_VoiceClone_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 文本/翻译类 (保持不变)
    "LLM_Translator": "🧠 LLM 智能翻译 (Qwen)",
    "LLM_Chat": "💬 LLM 智能对话 (Qwen)",
    
    # 音频类 (名称已修改：LLM -> Qwen)
    "Qwen_TTS": "🔊 Qwen 语音合成 (CustomVoice)",
    "Qwen_TTS_VoiceDesign": "🔊 Qwen 语音设计 (VoiceDesign)",
    "Qwen_TTS_VoiceClone": "🔊 Qwen 语音克隆 (VoiceClone)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]