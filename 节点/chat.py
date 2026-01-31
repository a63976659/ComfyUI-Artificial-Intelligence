import torch
import random
import numpy as np
from .utils import get_installed_models, load_config, save_config, load_llm_model

class LLM_Chat_Node:
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
                "提示词": ("STRING", {"multiline": True, "default": "你好，请介绍一下你自己。", "placeholder": "在此输入对话内容..."}),
                "模型名称": (all_models, {"default": default_model}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
                "温度_创造性": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}), 
                "Top_P_采样率": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "最大生成长度": ("INT", {"default": 2048, "min": 64, "max": 8192}),
                "自动下载模型": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "系统指令": ("STRING", {"multiline": True, "default": "你是一个有用的AI助手。"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回复内容",)
    FUNCTION = "chat"
    CATEGORY = "💬 AI人工智能"
    DESCRIPTION = "基于本地大模型的智能对话节点。支持随机种子控制、温度调整和自动模型下载。"

    def chat(self, 提示词, 模型名称, 随机种子, 温度_创造性, Top_P_采样率, 最大生成长度, 自动下载模型, 系统指令):
        # 1. 保存配置
        save_config(模型名称)

        # 2. 设置随机种子
        if 随机种子 is not None:
            torch.manual_seed(随机种子)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(随机种子)
            np.random.seed(随机种子)
            random.seed(随机种子)

        # 3. 加载模型
        tokenizer, model = load_llm_model(模型名称, self.device, 自动下载模型)

        # 4. 构建对话
        messages = [
            {"role": "system", "content": 系统指令},
            {"role": "user", "content": 提示词}
        ]
        
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 5. 推理
        model_inputs = tokenizer([text_input], return_tensors="pt").to(self.device)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=最大生成长度,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=温度_创造性,
            top_p=Top_P_采样率
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return (response,)