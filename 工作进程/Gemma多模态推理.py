# -*- coding: utf-8 -*-
"""
Gemma 4 隔离进程推理脚本
运行环境: gemma_env (transformers>=5.14)，不要在 ComfyUI 主进程中导入本文件。
"""
import re

from 协议 import log, send, main_loop

# 全局模型缓存 (进程内只驻留一份)
_MODEL = None
_PROCESSOR = None
_LOADED_KEY = None


def load_model(model_path, quant):
    """按 (路径, 量化方式) 加载/复用模型，配置变化时重新加载"""
    global _MODEL, _PROCESSOR, _LOADED_KEY
    import torch

    key = (model_path, quant)
    if _LOADED_KEY == key and _MODEL is not None:
        return

    if _MODEL is not None:
        log("模型配置变化，卸载旧模型...")
        _MODEL = None
        _PROCESSOR = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    from transformers import AutoProcessor, AutoModelForMultimodalLM

    log(f"加载模型: {model_path} (量化: {quant})")
    # bitsandbytes 量化仅支持 NVIDIA CUDA，其它设备 (AMD/Intel/Apple/CPU) 自动回退不量化
    if quant in ("4bit", "8bit") and not torch.cuda.is_available():
        log(f"当前设备无 CUDA，{quant} 量化不可用，自动回退为 bf16 (显存/内存需求会显著增加)")
        quant = "bf16"
    kwargs = {"device_map": "auto"}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif quant == "8bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:  # bf16 / auto
        kwargs["dtype"] = "auto"

    _PROCESSOR = AutoProcessor.from_pretrained(model_path)
    _MODEL = AutoModelForMultimodalLM.from_pretrained(model_path, **kwargs)
    _LOADED_KEY = key
    log("模型加载完成")


def _apply_template(messages, enable_thinking, visual_token_budget):
    """构建输入。enable_thinking 为模板变量; 视觉 token 预算对应 processor 的
    max_soft_tokens (合法值 70/140/280/560/1120)，必需通过 processor_kwargs 嵌套字典传入。"""
    base = dict(
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not visual_token_budget:
        return _PROCESSOR.apply_chat_template(messages, **base)

    try:
        return _PROCESSOR.apply_chat_template(
            messages,
            processor_kwargs={"images_kwargs": {"max_soft_tokens": visual_token_budget}},
            **base,
        )
    except (TypeError, ValueError) as e:
        log(f"警告: 视觉Token预算未生效 ({e})，已按模型默认分辨率处理")
        return _PROCESSOR.apply_chat_template(messages, **base)


def _parse_output(raw, prefix_ids):
    """解析输出，分离思考过程与正式回复。优先用官方 parse_response，失败则正则兜底。"""
    try:
        parsed = _PROCESSOR.parse_response(raw, prefix=prefix_ids)
        if isinstance(parsed, dict):
            thinking = parsed.get("thinking") or parsed.get("thought") or ""
            content = parsed.get("content") or parsed.get("response") or parsed.get("text") or ""
            if content:
                return str(thinking).strip(), str(content).strip()
    except Exception as e:
        log(f"parse_response 不可用，使用正则解析: {e}")

    thinking = ""
    m = re.search(r"<\|channel>thought\n(.*?)<channel\|>", raw, re.S)
    if m:
        thinking = m.group(1).strip()
        raw = raw[m.end():]
    # 清理残留的特殊 token
    content = re.sub(r"<\|[^<>]*?\|>|<\|[^<>]*?>|<[^<>]*?\|>", "", raw)
    content = re.sub(r"<end_of_turn>|<start_of_turn>|<eos>|<bos>|</?s>", "", content).strip()
    return thinking, content


def handle_generate(req):
    import torch

    load_model(req["model_path"], req.get("quant", "4bit"))

    seed = int(req.get("seed", 0)) & 0xffffffff
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 多模态内容: 官方最佳实践要求图像/音频放在文本之前
    content = []
    image_paths = req.get("image_paths") or ([req["image_path"]] if req.get("image_path") else [])
    if image_paths:
        from PIL import Image
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            content.append({"type": "image", "image": img})
    if req.get("audio_path"):
        # 用 librosa 预解码为 numpy 数组再传给 processor，避免 transformers 内部走
        # torchaudio/torchcodec 解码 (本机缺 libtorchcodec DLL 会崩溃)；librosa 走 soundfile。
        import librosa
        target_sr = 16000
        fe = getattr(_PROCESSOR, "feature_extractor", None)
        if fe is not None and getattr(fe, "sampling_rate", None):
            target_sr = int(fe.sampling_rate)
        audio_array, _ = librosa.load(req["audio_path"], sr=target_sr, mono=True)
        content.append({"type": "audio", "audio": audio_array})
    content.append({"type": "text", "text": req["prompt"]})

    messages = []
    if req.get("system"):
        messages.append({"role": "system", "content": req["system"]})
    messages.append({"role": "user", "content": content})

    inputs = _apply_template(
        messages,
        enable_thinking=bool(req.get("enable_thinking", False)),
        visual_token_budget=int(req.get("visual_token_budget", 0) or 0),
    ).to(_MODEL.device)
    # 量化/半精度下非量化层 (如视觉 LayerNorm) 为 bf16/fp16，
    # 浮点输入 (pixel_values 等) 需同步精度，否则 layer_norm 报 dtype 不匹配
    if _MODEL.dtype in (torch.bfloat16, torch.float16):
        inputs = inputs.to(_MODEL.dtype)
    input_len = inputs["input_ids"].shape[-1]

    log(f"开始推理 (输入 {input_len} tokens, seed={seed})...")
    # 官方推荐采样参数: temperature=1.0, top_p=0.95, top_k=64
    outputs = _MODEL.generate(
        **inputs,
        max_new_tokens=int(req.get("max_new_tokens", 1024)),
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    raw = _PROCESSOR.decode(outputs[0][input_len:], skip_special_tokens=False)
    thinking, text = _parse_output(raw, inputs["input_ids"])
    log("推理完成")
    return {"ok": True, "content": text, "thinking": thinking}


if __name__ == "__main__":
    main_loop({"generate": handle_generate})
