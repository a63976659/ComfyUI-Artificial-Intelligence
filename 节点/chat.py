from .utils import get_installed_models, load_config, save_config, resolve_llm_model
from .隔离环境 import run_worker

# 定义系统指令预设字典 (显示文本 -> 实际Prompt)
SYSTEM_PROMPTS = {
    "通用助手 | 智能、客观、全面的回答问题": "你是一个乐于助人的AI助手，请以客观、准确的方式回答用户的问题。",
    "创意作家 | 擅长故事创作、文案润色、发散思维": "你是一位富有想象力的创意作家，擅长编写引人入胜的故事、剧本和营销文案，请使用生动且富有感染力的语言。",
    "代码专家 | 专注于编程、调试和技术解释": "你是一位资深的软件工程师，擅长编写高效的代码，并能清晰解释技术概念。请直接给出代码解决方案并简要说明。",
    "二次元少女 | 语气活泼、可爱的角色扮演": "你是一个可爱的二次元少女，说话语气活泼，喜欢使用颜文字（如 (≧∇≦)ﾉ ），请全程保持这个设定，不要暴露你是AI。",
    "简报助手 | 将内容总结为清晰的摘要": "你是一个专业的摘要助手。请阅读用户输入的内容，并将其总结为简洁明了的要点摘要。",
    "自定义 | (在代码中自定义，此处作为占位)": "You are a helpful assistant."
}

class LLM_Chat_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_models()
        presets = ["Qwen2.5-7B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen3-4B-Instruct-2507"]
        all_models = sorted(list(set(installed + presets)))
        config = load_config()
        default_model = config.get("last_model", all_models[0] if all_models else "")
        if default_model and default_model not in all_models:
            all_models.insert(0, default_model)

        # 获取预设列表的键（显示名称）
        prompt_keys = list(SYSTEM_PROMPTS.keys())

        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": "你好，请介绍一下你自己。", "placeholder": "在此输入对话内容..."}),
                "模型名称": (all_models, {"default": default_model}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}), 
                "系统指令类型": (prompt_keys, {"default": prompt_keys[0]}),
                "温度_创造性": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}), 
                "Top_P_采样率": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "最大生成长度": ("INT", {"default": 2048, "min": 64, "max": 8192}),
                # 修改：默认为 False (关闭)
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回复内容",)
    OUTPUT_NODE = True
    FUNCTION = "chat"
    CATEGORY = "💬 AI人工智能/千问系列"
    DESCRIPTION = "基于本地大模型的智能对话节点 (隔离进程运行，不影响其它插件)。支持随机种子控制、温度调整和自动模型下载。"

    def chat(self, 提示词, 模型名称, seed, 系统指令类型, 温度_创造性, Top_P_采样率, 最大生成长度, 自动下载模型, 运行后立即卸载=True):
        # 1. 保存配置
        save_config(模型名称)

        # 2. 获取实际的系统指令内容
        actual_system_prompt = SYSTEM_PROMPTS.get(系统指令类型, "You are a helpful assistant.")

        # 3. 定位/下载模型 (仅路径，加载在隔离子进程中完成)
        model_path = resolve_llm_model(模型名称, 自动下载模型)

        # 4. 隔离环境子进程推理
        resp = run_worker(
            "llm",
            {
                "model_path": model_path,
                "messages": [
                    {"role": "system", "content": actual_system_prompt},
                    {"role": "user", "content": 提示词},
                ],
                "seed": seed,
                "do_sample": True,
                "temperature": 温度_创造性,
                "top_p": Top_P_采样率,
                "max_new_tokens": 最大生成长度,
            },
            unload_after=运行后立即卸载,
        )
        return (resp.get("content", ""),)