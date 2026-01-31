from .节点.translator import LLM_Translator_Node
from .节点.chat import LLM_Chat_Node

NODE_CLASS_MAPPINGS = {
    "LLM_Translator": LLM_Translator_Node,
    "LLM_Chat": LLM_Chat_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_Translator": "🧠 LLM 智能翻译 (Qwen)",
    "LLM_Chat": "💬 LLM 智能对话 (Qwen)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]