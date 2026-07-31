# -*- coding: utf-8 -*-
"""
Qwen 文本模型 (对话/翻译) 隔离进程推理脚本
运行环境: llm_env (transformers==4.57.3)，不要在 ComfyUI 主进程中导入本文件。
兼容多家国产模型: prompt_style 支持 qwen(默认) / hunyuan / seedx，
并支持 stream 流式生成 (逐段通过流式协议发回主进程)。
"""
from 协议 import log, main_loop, send_stream

# 全局模型缓存 (进程内只驻留一份)
_MODEL = None
_TOKENIZER = None
_LOADED_PATH = None


def load_model(model_path):
    """按路径加载/复用模型，路径变化时重新加载"""
    global _MODEL, _TOKENIZER, _LOADED_PATH
    import torch

    if _LOADED_PATH == model_path and _MODEL is not None:
        return

    if _MODEL is not None:
        log("模型变化，卸载旧模型...")
        _MODEL = None
        _TOKENIZER = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"加载模型: {model_path}")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    _LOADED_PATH = model_path
    log("模型加载完成")


def handle_generate(req):
    import torch

    load_model(req["model_path"])

    seed = req.get("seed")
    if seed is not None:
        import random
        import numpy as np
        seed = int(seed) & 0xffffffff
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    prompt_style = req.get("prompt_style", "qwen")

    # === 按模型风格构造输入 ===
    if prompt_style == "seedx":
        # Seed-X: Mistral 架构、无 chat template，直接使用原始 prompt
        text_input = req["prompt"]
    else:
        # qwen / hunyuan 均走各自的 chat template
        messages = req["messages"]
        text_input = _TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    model_inputs = _TOKENIZER([text_input], return_tensors="pt").to(_MODEL.device)

    # === 按模型风格配置生成参数 ===
    gen_kwargs = {
        "max_new_tokens": int(req.get("max_new_tokens", 1024)),
        "pad_token_id": _TOKENIZER.eos_token_id,
    }
    if prompt_style == "hunyuan":
        # Hunyuan-MT 官方推荐采样参数
        gen_kwargs.update(
            do_sample=True,
            top_k=20,
            top_p=0.6,
            temperature=0.7,
            repetition_penalty=1.05,
        )
    elif prompt_style == "seedx":
        # Seed-X 翻译保持贪心解码
        pass
    elif req.get("do_sample"):
        gen_kwargs.update(
            do_sample=True,
            temperature=float(req.get("temperature", 0.7)),
            top_p=float(req.get("top_p", 0.9)),
        )

    log(f"开始推理 (输入 {model_inputs.input_ids.shape[-1]} tokens, 风格 {prompt_style})...")

    # === 流式生成: 逐段通过流式协议发回主进程 ===
    if req.get("stream"):
        import threading
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            _TOKENIZER, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer
        thread = threading.Thread(
            target=_MODEL.generate,
            args=(model_inputs.input_ids,),
            kwargs=gen_kwargs,
            daemon=True,
        )
        thread.start()

        pieces = []
        for piece in streamer:
            if piece:
                pieces.append(piece)
                send_stream({"delta": piece})
        thread.join()
        response = "".join(pieces)
        log("流式推理完成")
        return {"ok": True, "content": response}

    # === 非流式路径 (保持原逻辑) ===
    generated_ids = _MODEL.generate(model_inputs.input_ids, **gen_kwargs)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = _TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
    log("推理完成")
    return {"ok": True, "content": response}


if __name__ == "__main__":
    main_loop({"generate": handle_generate})
