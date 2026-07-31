# -*- coding: utf-8 -*-
"""
Qwen3-TTS 隔离进程推理脚本
运行环境: tts_env (transformers==4.57.3 + qwen-tts)，不要在 ComfyUI 主进程中导入本文件。
音频数据通过临时 .npy 文件与主进程交换 (请求中的 output_dir / ref_audio_path)。
"""
import os
import uuid

from 协议 import log, main_loop

# 全局模型缓存 (进程内只驻留一份)
_MODEL = None
_LOADED_PATH = None


def load_model(model_path):
    """按路径加载/复用模型，路径变化时重新加载"""
    global _MODEL, _LOADED_PATH
    import torch

    if _LOADED_PATH == model_path and _MODEL is not None:
        return

    if _MODEL is not None:
        log("模型变化，卸载旧模型...")
        _MODEL = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    from qwen_tts import Qwen3TTSModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log(f"加载模型: {model_path} -> {device}")
    _MODEL = Qwen3TTSModel.from_pretrained(model_path, device_map=device, torch_dtype=dtype)
    _LOADED_PATH = model_path
    log("模型加载完成")


def _set_seed(seed):
    if seed is None:
        return
    import random
    import numpy as np
    import torch
    seed = int(seed) & 0xffffffff
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def handle_generate(req):
    """按段落生成语音，每段波形存为 .npy 返回路径列表 (停顿段生成静音)"""
    import numpy as np

    load_model(req["model_path"])
    _set_seed(req.get("seed"))

    mode = req["mode"]  # custom_voice / voice_design / voice_clone
    language = req.get("language", "Auto")
    instruct = req.get("instruct") or None
    gen_kwargs = req.get("gen_kwargs", {})
    output_dir = req["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 语音克隆: 加载参考音频特征并构建 voice_prompt
    voice_prompt = None
    if mode == "voice_clone":
        ref_wav = np.load(req["ref_audio_path"])
        x_vector_only = bool(req.get("x_vector_only", False))
        ref_text = req.get("ref_text") or None
        log(f"提取参考音频特征... 模式={'X-Vector(极速)' if x_vector_only else 'ICL(高质量)'}")
        voice_prompt = _MODEL.create_voice_clone_prompt(
            ref_audio=(ref_wav, int(req.get("ref_sample_rate", 16000))),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )

    segment_paths = []
    sr = 24000
    pending_pauses = []  # 先记录停顿时长，等确定采样率后统一生成静音

    for seg_type, content in req["segments"]:
        if seg_type == "pause":
            if content > 0:
                path = os.path.join(output_dir, f"tts_{uuid.uuid4().hex}.npy")
                pending_pauses.append((len(segment_paths), float(content), path))
                segment_paths.append(path)
            continue

        log(f"合成: {str(content)[:40]}...")
        if mode == "custom_voice":
            wavs, current_sr = _MODEL.generate_custom_voice(
                text=[content], language=[language], speaker=[req["speaker"]],
                instruct=[instruct] if instruct else None, **gen_kwargs
            )
        elif mode == "voice_design":
            wavs, current_sr = _MODEL.generate_voice_design(
                text=[content], language=[language],
                instruct=[instruct], **gen_kwargs
            )
        else:  # voice_clone
            wavs, current_sr = _MODEL.generate_voice_clone(
                text=[content], language=[language],
                voice_clone_prompt=voice_prompt,
                instruct=[instruct] if instruct else None, **gen_kwargs
            )
        sr = current_sr
        if len(wavs) > 0:
            wav = wavs[0].squeeze() if wavs[0].ndim > 1 else wavs[0]
            path = os.path.join(output_dir, f"tts_{uuid.uuid4().hex}.npy")
            np.save(path, np.asarray(wav, dtype=np.float32))
            segment_paths.append(path)

    # 用最终采样率补写静音段
    for _, duration, path in pending_pauses:
        np.save(path, np.zeros(int(duration * sr), dtype=np.float32))

    log("合成完成")
    return {"ok": True, "segment_paths": segment_paths, "sample_rate": int(sr)}


if __name__ == "__main__":
    main_loop({"generate": handle_generate})
