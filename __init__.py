import logging

logger = logging.getLogger("ComfyUI-AI")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ================= 1. 后端 API 容错导入 =================
try:
    from . import server_api
    logger.info("[ComfyUI-AI] 后端API加载成功 [OK]")
except Exception as e:
    logger.warning(f"[ComfyUI-AI] 后端API加载失败，文件浏览和媒体预览功能不可用: {e}")

# ================= 2. 节点容错导入 =================

# 翻译节点
try:
    from .节点.translator import LLM_Translator_Node
    NODE_CLASS_MAPPINGS["LLM_Translator"] = LLM_Translator_Node
    NODE_DISPLAY_NAME_MAPPINGS["LLM_Translator"] = "🧠 LLM 智能翻译 (Qwen)"
    logger.info("[ComfyUI-AI] 翻译节点加载成功 [OK]")
except Exception as e:
    logger.warning(f"[ComfyUI-AI] 翻译节点加载失败: {e}")

# 对话节点
try:
    from .节点.chat import LLM_Chat_Node
    NODE_CLASS_MAPPINGS["LLM_Chat"] = LLM_Chat_Node
    NODE_DISPLAY_NAME_MAPPINGS["LLM_Chat"] = "💬 LLM 智能对话 (Qwen)"
    logger.info("[ComfyUI-AI] 对话节点加载成功 [OK]")
except Exception as e:
    logger.warning(f"[ComfyUI-AI] 对话节点加载失败: {e}")

# TTS语音合成节点
try:
    from .节点.tts import Qwen_TTS_Node, Qwen_TTS_VoiceDesign_Node, Qwen_TTS_VoiceClone_Node
    NODE_CLASS_MAPPINGS["Qwen_TTS"] = Qwen_TTS_Node
    NODE_CLASS_MAPPINGS["Qwen_TTS_VoiceDesign"] = Qwen_TTS_VoiceDesign_Node
    NODE_CLASS_MAPPINGS["Qwen_TTS_VoiceClone"] = Qwen_TTS_VoiceClone_Node
    NODE_DISPLAY_NAME_MAPPINGS["Qwen_TTS"] = "🔊 Qwen 语音合成 (CustomVoice)"
    NODE_DISPLAY_NAME_MAPPINGS["Qwen_TTS_VoiceDesign"] = "🔊 Qwen 语音设计 (VoiceDesign)"
    NODE_DISPLAY_NAME_MAPPINGS["Qwen_TTS_VoiceClone"] = "🔊 Qwen 语音克隆 (VoiceClone)"
    logger.info("[ComfyUI-AI] TTS语音合成节点加载成功 [OK]")
except Exception as e:
    logger.warning(f"[ComfyUI-AI] TTS语音合成节点加载失败: {e}")

# 音频加载节点
try:
    from .节点.加载音频 import 批量加载音频_Node, 加载音频_Node
    NODE_CLASS_MAPPINGS["批量加载音频"] = 批量加载音频_Node
    NODE_CLASS_MAPPINGS["加载音频"] = 加载音频_Node
    NODE_DISPLAY_NAME_MAPPINGS["批量加载音频"] = "📂 批量加载音频 (Batch Loader)"
    NODE_DISPLAY_NAME_MAPPINGS["加载音频"] = "🎵 加载音频 (Audio Loader)"
    logger.info("[ComfyUI-AI] 音频节点加载成功 [OK]")
except Exception as e:
    logger.warning(f"[ComfyUI-AI] 音频节点加载失败: {e}")

# 视频加载节点
try:
    from .节点.加载视频 import 加载视频_Node, 裁剪视频_Node
    NODE_CLASS_MAPPINGS["加载视频"] = 加载视频_Node
    NODE_CLASS_MAPPINGS["裁剪视频"] = 裁剪视频_Node
    NODE_DISPLAY_NAME_MAPPINGS["加载视频"] = "🎬 加载视频 (Load Video)"
    NODE_DISPLAY_NAME_MAPPINGS["裁剪视频"] = "✂️ 裁剪视频 (Crop Video)"
    logger.info("[ComfyUI-AI] 视频节点加载成功 [OK]")
except Exception as e:
    logger.warning(f"[ComfyUI-AI] 视频节点加载失败: {e}")

# ================= 3. 加载总结 =================
WEB_DIRECTORY = "./web"

loaded = len(NODE_CLASS_MAPPINGS)
total = 9
if loaded == total:
    logger.info(f"[ComfyUI-AI] 所有 {total} 个节点加载成功 [OK]")
elif loaded > 0:
    logger.warning(f"[ComfyUI-AI] 部分节点加载成功 ({loaded}/{total})，请检查上方警告信息并安装缺失依赖")
else:
    logger.error(f"[ComfyUI-AI] 所有节点加载失败，请运行: pip install -r requirements.txt")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]