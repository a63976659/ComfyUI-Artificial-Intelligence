from .utils import get_installed_models, load_config, save_config, resolve_llm_model
from .隔离环境 import run_worker

class LLM_Translator_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_models()
        presets = ["Qwen2.5-7B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen3-4B-Instruct-2507"]
        all_models = sorted(list(set(installed + presets)))
        config = load_config()
        default_model = config.get("last_model", all_models[0] if all_models else "")
        
        if default_model and default_model not in all_models:
            all_models.insert(0, default_model)

        # 完整的目标语言列表
        target_languages = [
            "中文", "英文", "日文", "韩文", "法文", "德文",
            "西班牙语", "俄语", "阿拉伯语", "葡萄牙语（包括巴西葡萄牙语）", "意大利语", 
            "泰语", "印地语", "越南语", "印尼语", "荷兰语", "土耳其语", "阿姆哈拉语", 
            "希腊语", "波斯语（伊朗语）", "阿尔巴尼亚语", "乌尔都语", "塞尔维亚语", 
            "立陶宛语", "芬兰语", "冰岛语", "马来语", "保加利亚语", 
            "哥伦比亚西班牙语（特定方言）", "新西兰英语（含地方表达）"
        ]

        return {
            "required": {
                "文本内容": ("STRING", {"multiline": True, "default": "一个女孩在雨中"}),
                "模型名称": (all_models, {"default": default_model}),
                "目标语言": (target_languages, {"default": "英文"}),
                # 新增：提示词润色开关
                "提示词润色": ("BOOLEAN", {"default": False}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "最大生成长度": ("INT", {"default": 1024}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("翻译结果",)
    FUNCTION = "translate"
    CATEGORY = "💬 AI人工智能/千问系列"
    DESCRIPTION = "使用本地LLM模型进行多语言翻译 (隔离进程运行，不影响其它插件)。开启'提示词润色'可自动丰富细节，适合绘画Prompt生成。"

    def translate(self, 文本内容, 模型名称, 目标语言, 提示词润色, 自动下载模型, 最大生成长度, 运行后立即卸载=True):
        save_config(模型名称)
        
        # === 核心逻辑修改：根据开关切换隐藏指令 ===
        
        # 指令 1: 纯净翻译 (关闭润色时使用)
        instruction_1 = "You are a professional translator. Translate the following text directly without explanation."
        
        # 指令 2: 翻译 + 润色/美化 (开启润色时使用)
        instruction_2 = (
            "You are a professional prompt engineer and translator. Your task is to translate the user's input into the target language. "
            "CRITICAL: You must also refine, beautify, and add descriptive details (lighting, texture, atmosphere, composition) "
            "to make the text vivid and high-quality, suitable for AI art generation. "
            "Output ONLY the final result without explanation or conversational filler."
        )

        # 根据开关选择指令
        if 提示词润色:
            system_instruction = instruction_2
        else:
            system_instruction = instruction_1

        # ==========================================

        # 定位/下载模型 (仅路径，加载在隔离子进程中完成)
        model_path = resolve_llm_model(模型名称, 自动下载模型)
        
        # 语言映射字典
        lang_map = {
            "中文": "Chinese", "英文": "English", "日文": "Japanese", 
            "韩文": "Korean", "法文": "French", "德文": "German",
            "西班牙语": "Spanish", "俄语": "Russian", "阿拉伯语": "Arabic", 
            "葡萄牙语（包括巴西葡萄牙语）": "Portuguese (including Brazilian Portuguese)", 
            "意大利语": "Italian", "泰语": "Thai", "印地语": "Hindi", 
            "越南语": "Vietnamese", "印尼语": "Indonesian", "荷兰语": "Dutch", 
            "土耳其语": "Turkish", "阿姆哈拉语": "Amharic", "希腊语": "Greek", 
            "波斯语（伊朗语）": "Persian (Farsi)", "阿尔巴尼亚语": "Albanian", 
            "乌尔都语": "Urdu", "塞尔维亚语": "Serbian", "立陶宛语": "Lithuanian", 
            "芬兰语": "Finnish", "冰岛语": "Icelandic", "马来语": "Malay", 
            "保加利亚语": "Bulgarian", 
            "哥伦比亚西班牙语（特定方言）": "Colombian Spanish (Specific Dialect)", 
            "新西兰英语（含地方表达）": "New Zealand English (Including local expressions)"
        }
        target_lang_en = lang_map.get(目标语言, 目标语言)

        # 隔离环境子进程推理 (翻译保持确定性输出，不开启采样)
        resp = run_worker(
            "llm",
            {
                "model_path": model_path,
                "messages": [
                    {"role": "system", "content": f"{system_instruction} Target Language: {target_lang_en}."},
                    {"role": "user", "content": 文本内容},
                ],
                "max_new_tokens": 最大生成长度,
            },
            unload_after=运行后立即卸载,
        )
        return (resp.get("content", ""),)