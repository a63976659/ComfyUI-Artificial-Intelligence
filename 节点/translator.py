import torch
from .utils import get_installed_models, load_config, save_config, load_llm_model

class LLM_Translator_Node:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_models()
        # 已添加 Qwen3-4B-Instruct-2507
        presets = ["Qwen2.5-7B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen3-4B-Instruct-2507"]
        all_models = sorted(list(set(installed + presets)))
        config = load_config()
        default_model = config.get("last_model", all_models[0] if all_models else "")
        
        if default_model and default_model not in all_models:
            all_models.insert(0, default_model)

        return {
            "required": {
                "文本内容": ("STRING", {"multiline": True, "default": "你好，世界"}),
                "模型名称": (all_models, {"default": default_model}),
                "目标语言": (["中文", "英文", "日文", "韩文", "法文", "德文"], {"default": "中文"}),
                "自动下载模型": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "系统指令": ("STRING", {"multiline": True, "default": "你是一个专业的翻译助手。"}),
                "最大生成长度": ("INT", {"default": 1024}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("翻译结果",)
    FUNCTION = "translate"
    CATEGORY = "💬 AI人工智能"
    DESCRIPTION = "使用本地LLM模型进行多语言翻译。包含自动下载模型功能。"

    def translate(self, 文本内容, 模型名称, 目标语言, 自动下载模型, 系统指令, 最大生成长度):
        save_config(模型名称)
        
        # 简单处理：尝试自动补全 repo_id 用于下载
        # 如果模型名称中不包含 "/" 且包含 "Qwen"，则尝试加上 "Qwen/" 前缀
        # 注意：这只是为了猜测下载路径，如果您的模型不在 Qwen 官方仓库下，请手动下载
        download_repo_id = 模型名称
        if 自动下载模型 and "Qwen" in 模型名称 and "/" not in 模型名称:
             download_repo_id = f"Qwen/{模型名称}"

        tokenizer, model = load_llm_model(模型名称, self.device, 自动下载模型)
        
        lang_map = {
            "中文": "Chinese", "英文": "English", "日文": "Japanese", 
            "韩文": "Korean", "法文": "French", "德文": "German"
        }
        target_lang_en = lang_map.get(目标语言, 目标语言)

        messages = [
            {"role": "system", "content": f"{系统指令} Target Language: {target_lang_en}."},
            {"role": "user", "content": 文本内容}
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(self.device)
        
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=最大生成长度, pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return (tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0],)