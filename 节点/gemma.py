# -*- coding: utf-8 -*-
"""
Gemma 4 多模态节点 (隔离进程方案)
由于 qwen-tts 锁定 transformers==4.57.3，而 Gemma 4 需要 transformers>=5.14，
本节点通过独立虚拟环境 gemma_env 中的子进程 (工作进程/Gemma多模态推理.py) 完成推理:
- 主进程不导入新版 transformers，对其它节点零影响
- 子进程退出即彻底释放显存 (运行后立即卸载)
- 常驻模式下空闲超时自动退出
环境与子进程的通用管理见 节点/隔离环境.py
"""
import os
import uuid

import numpy as np
from PIL import Image

import folder_paths
from .utils import LLM_MODELS_DIR, HAS_MODELSCOPE
from .chat import SYSTEM_PROMPTS
from .隔离环境 import run_worker
from huggingface_hub import snapshot_download as hf_snapshot_download

if HAS_MODELSCOPE:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

DEFAULT_GEMMA_MODEL = "gemma-4-12B-it"

# ================= 模型下载 =================

def resolve_gemma_model(model_name, auto_download, source):
    """定位本地模型目录，不存在时按下载源自动下载"""
    target_folder = model_name.split("/")[-1] if "/" in model_name else model_name
    repo_id = model_name if "/" in model_name else f"google/{model_name}"

    for p in (os.path.join(LLM_MODELS_DIR, model_name), os.path.join(LLM_MODELS_DIR, target_folder)):
        if os.path.exists(p) and any(f.endswith(".safetensors") for f in os.listdir(p)):
            return p

    if not auto_download:
        raise FileNotFoundError(
            f"未找到模型 {model_name} (应位于 models/LLM/{target_folder})，"
            f"请开启\"自动下载模型\"或手动下载。"
        )

    download_path = os.path.join(LLM_MODELS_DIR, target_folder)
    if source == "ModelScope":
        if not HAS_MODELSCOPE:
            raise ImportError("请先安装 modelscope: pip install modelscope")
        print(f"[Gemma] 从 ModelScope 下载: {repo_id} -> {download_path}")
        ms_snapshot_download(model_id=repo_id, local_dir=download_path)
    elif source == "HF Mirror":
        print(f"[Gemma] 从 HF Mirror 下载: {repo_id} -> {download_path}")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        hf_snapshot_download(repo_id=repo_id, local_dir=download_path, resume_download=True, max_workers=4)
    else:  # HuggingFace 官方
        print(f"[Gemma] 从 HuggingFace 下载: {repo_id} -> {download_path}")
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        hf_snapshot_download(repo_id=repo_id, local_dir=download_path, resume_download=True, max_workers=4)
    return download_path


def get_installed_gemma_models():
    """扫描 models/LLM 下的 gemma-4 系列模型"""
    if not os.path.exists(LLM_MODELS_DIR):
        return []
    return sorted(
        d for d in os.listdir(LLM_MODELS_DIR)
        if os.path.isdir(os.path.join(LLM_MODELS_DIR, d)) and "gemma-4" in d.lower()
    )

# ================= 节点定义 =================

# 任务预设 (显示文本 -> 提示词, None 表示使用自定义问题)
TASK_PROMPTS = {
    "反推提示词 | 生成SD/Flux英文提示词": (
        "Describe this image as a detailed image-generation prompt in English for "
        "Stable Diffusion / Flux. Output only comma-separated tags and short phrases "
        "covering subject, appearance, style, lighting, composition and quality. "
        "Do not write full sentences or explanations."
    ),
    "详细描述 | 中文详细描述图片内容": "请用中文详细描述这张图片的内容，包括主体、背景、色彩、光线、构图和整体氛围。",
    "简短标题 | 一句话概括图片": "请用一句简短的中文概括这张图片的内容，不超过20个字。只输出这句话。",
    "OCR提取 | 提取图中所有文字": (
        "Extract all visible text from this image exactly as written. "
        "Preserve the original language and reading order. Output only the extracted text."
    ),
    "自定义问题 | 使用下方自定义问题提问": None,
}

VISUAL_TOKEN_BUDGETS = ["70", "140", "280", "560", "1120"]


# ================= 临时文件辅助 =================

def _temp_path(suffix):
    tmp_dir = folder_paths.get_temp_directory()
    os.makedirs(tmp_dir, exist_ok=True)
    return os.path.join(tmp_dir, f"gemma_{uuid.uuid4().hex}{suffix}")


def _save_temp_image(img_tensor):
    """单帧 [H,W,C] float 张量存为临时 PNG，返回路径"""
    img_np = (np.clip(img_tensor.cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
    path = _temp_path(".png")
    Image.fromarray(img_np).save(path)
    return path


def _save_temp_audio(audio):
    """ComfyUI AUDIO ({'waveform','sample_rate'}) 存为临时 WAV，返回路径。
    使用 soundfile 写出，避免 torchaudio 的 torchcodec 后端依赖。"""
    import soundfile as sf
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    waveform = audio["waveform"]
    if waveform.dim() == 3:
        waveform = waveform[0]
    path = _temp_path(".wav")
    # soundfile 需要 (frames, channels)，waveform 为 (channels, frames)
    sf.write(path, waveform.t().cpu().numpy(), int(audio["sample_rate"]))
    return path


def _cleanup_files(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _run_gemma(模型名称, 自动下载模型, 下载源, payload, unload_after):
    """公共流程: 定位/下载模型 -> 隔离环境子进程推理"""
    payload["model_path"] = resolve_gemma_model(模型名称, 自动下载模型, 下载源)
    resp = run_worker("gemma", payload, unload_after=unload_after)
    return (resp.get("content", ""), resp.get("thinking", ""))


class Gemma_Image_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_gemma_models()
        all_models = sorted(set(installed + [DEFAULT_GEMMA_MODEL]))
        task_keys = list(TASK_PROMPTS.keys())
        return {
            "required": {
                "图像": ("IMAGE",),
                "任务类型": (task_keys, {"default": task_keys[0]}),
                "自定义问题": ("STRING", {"multiline": True, "default": "", "placeholder": "任务类型选\"自定义问题\"时在此输入..."}),
                "模型名称": (all_models, {"default": DEFAULT_GEMMA_MODEL}),
                "量化方式": (["4bit", "8bit", "bf16"], {"default": "4bit"}),
                "视觉Token预算": (VISUAL_TOKEN_BUDGETS, {"default": "560"}),
                "最大生成长度": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "思考模式": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HF Mirror", "HuggingFace"], {"default": "ModelScope"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("回复内容", "思考过程")
    FUNCTION = "analyze"
    CATEGORY = "💬 AI人工智能/谷歌系列"
    DESCRIPTION = (
        "基于 Gemma 4 的图像理解节点 (隔离进程运行，不影响其它节点)。"
        "支持反推提示词、图像描述、OCR 和自定义提问。"
        "OCR 建议用高视觉Token预算(1120)，打标/描述用中低预算。"
    )

    def analyze(self, 图像, 任务类型, 自定义问题, 模型名称, 量化方式, 视觉Token预算,
                最大生成长度, seed, 思考模式, 运行后立即卸载, 自动下载模型, 下载源):
        # 1. 确定提示词
        prompt = TASK_PROMPTS.get(任务类型)
        if prompt is None:
            prompt = 自定义问题.strip()
            if not prompt:
                raise ValueError("任务类型为\"自定义问题\"时，请在\"自定义问题\"中输入内容")

        # 2. 图像存为临时文件传给子进程 (取批次第一张)
        tmp_path = _save_temp_image(图像[0])

        # 3. 子进程推理
        try:
            return _run_gemma(
                模型名称, 自动下载模型, 下载源,
                {
                    "quant": 量化方式,
                    "image_path": tmp_path,
                    "prompt": prompt,
                    "enable_thinking": 思考模式,
                    "visual_token_budget": int(视觉Token预算),
                    "max_new_tokens": 最大生成长度,
                    "seed": seed,
                },
                unload_after=运行后立即卸载,
            )
        finally:
            _cleanup_files([tmp_path])


class Gemma_Chat_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_gemma_models()
        all_models = sorted(set(installed + [DEFAULT_GEMMA_MODEL]))
        prompt_keys = list(SYSTEM_PROMPTS.keys())
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": "你好，请介绍一下你自己。", "placeholder": "在此输入对话内容..."}),
                "系统指令类型": (prompt_keys, {"default": prompt_keys[0]}),
                "模型名称": (all_models, {"default": DEFAULT_GEMMA_MODEL}),
                "量化方式": (["4bit", "8bit", "bf16"], {"default": "4bit"}),
                "最大生成长度": ("INT", {"default": 2048, "min": 64, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "思考模式": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HF Mirror", "HuggingFace"], {"default": "ModelScope"}),
            },
            "optional": {
                "图像": ("IMAGE",),
                "音频": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("回复内容", "思考过程")
    FUNCTION = "chat"
    CATEGORY = "💬 AI人工智能/谷歌系列"
    DESCRIPTION = (
        "基于 Gemma 4 的多模态对话节点 (隔离进程运行)。"
        "可选接入图像/音频一起提问，都不接入时为纯文本对话。支持思考模式。"
    )

    def chat(self, 提示词, 系统指令类型, 模型名称, 量化方式, 最大生成长度, seed,
             思考模式, 运行后立即卸载, 自动下载模型, 下载源, 图像=None, 音频=None):
        if not 提示词.strip():
            raise ValueError("请输入提示词")

        tmp_files = []
        payload = {
            "quant": 量化方式,
            "prompt": 提示词,
            "system": SYSTEM_PROMPTS.get(系统指令类型, "You are a helpful assistant."),
            "enable_thinking": 思考模式,
            "max_new_tokens": 最大生成长度,
            "seed": seed,
        }
        if 图像 is not None:
            payload["image_path"] = _save_temp_image(图像[0])
            tmp_files.append(payload["image_path"])
        if 音频 is not None:
            payload["audio_path"] = _save_temp_audio(音频)
            tmp_files.append(payload["audio_path"])

        try:
            return _run_gemma(模型名称, 自动下载模型, 下载源, payload, unload_after=运行后立即卸载)
        finally:
            _cleanup_files(tmp_files)


# 音频理解任务预设 (显示文本 -> 提示词, None 表示使用自定义问题)
AUDIO_TASK_PROMPTS = {
    "转写文字 | 原样转写语音内容": (
        "Transcribe the following speech segment in its original language. "
        "Only output the transcription, with no extra explanations. "
        "When transcribing numbers, write the digits (e.g. 1.7 instead of one point seven)."
    ),
    "翻译中文 | 语音内容翻译成中文": "请听这段音频，并将其中的语音内容翻译成简体中文。只输出翻译结果。",
    "内容描述 | 描述音频中的声音内容": "请用中文描述这段音频的内容，包括语音、音乐、环境音等信息。",
    "自定义问题 | 使用下方自定义问题提问": None,
}


class Gemma_Audio_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_gemma_models()
        all_models = sorted(set(installed + [DEFAULT_GEMMA_MODEL]))
        task_keys = list(AUDIO_TASK_PROMPTS.keys())
        return {
            "required": {
                "音频": ("AUDIO",),
                "任务类型": (task_keys, {"default": task_keys[0]}),
                "自定义问题": ("STRING", {"multiline": True, "default": "", "placeholder": "任务类型选\"自定义问题\"时在此输入..."}),
                "模型名称": (all_models, {"default": DEFAULT_GEMMA_MODEL}),
                "量化方式": (["4bit", "8bit", "bf16"], {"default": "4bit"}),
                "最大生成长度": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HF Mirror", "HuggingFace"], {"default": "ModelScope"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回复内容",)
    FUNCTION = "analyze"
    CATEGORY = "💬 AI人工智能/谷歌系列"
    DESCRIPTION = (
        "基于 Gemma 4 的音频理解节点 (隔离进程运行)。"
        "支持语音转写、翻译成中文、音频内容描述和自定义提问，可直接连接\"加载音频\"节点。"
        "注意: 模型不支持中文语音 (官方音频基准注明 Excluding Chinese)，"
        "中文语音请使用 Qwen 语音识别 (ASR) 节点；音频最长 30 秒。"
    )

    def analyze(self, 音频, 任务类型, 自定义问题, 模型名称, 量化方式, 最大生成长度,
                seed, 运行后立即卸载, 自动下载模型, 下载源):
        prompt = AUDIO_TASK_PROMPTS.get(任务类型)
        if prompt is None:
            prompt = 自定义问题.strip()
            if not prompt:
                raise ValueError("任务类型为\"自定义问题\"时，请在\"自定义问题\"中输入内容")

        tmp_path = _save_temp_audio(音频)
        try:
            content, _ = _run_gemma(
                模型名称, 自动下载模型, 下载源,
                {
                    "quant": 量化方式,
                    "audio_path": tmp_path,
                    "prompt": prompt,
                    "max_new_tokens": 最大生成长度,
                    "seed": seed,
                },
                unload_after=运行后立即卸载,
            )
            return (content,)
        finally:
            _cleanup_files([tmp_path])


# 视频理解任务预设 (显示文本 -> 提示词, None 表示使用自定义问题)
VIDEO_TASK_PROMPTS = {
    "内容描述 | 中文描述视频内容": (
        "以下是同一段视频按时间顺序抽取的帧。请用中文描述这段视频的内容，"
        "包括主体、动作、场景变化和整体氛围。"
    ),
    "剧情摘要 | 总结视频剧情要点": (
        "以下是同一段视频按时间顺序抽取的帧。请用中文总结这段视频的剧情要点，按时间顺序列出。"
    ),
    "自定义问题 | 使用下方自定义问题提问": None,
}


class Gemma_Video_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_gemma_models()
        all_models = sorted(set(installed + [DEFAULT_GEMMA_MODEL]))
        task_keys = list(VIDEO_TASK_PROMPTS.keys())
        return {
            "required": {
                "图像序列": ("IMAGE",),
                "任务类型": (task_keys, {"default": task_keys[0]}),
                "自定义问题": ("STRING", {"multiline": True, "default": "", "placeholder": "任务类型选\"自定义问题\"时在此输入..."}),
                "抽帧数量": ("INT", {"default": 8, "min": 1, "max": 32}),
                "模型名称": (all_models, {"default": DEFAULT_GEMMA_MODEL}),
                "量化方式": (["4bit", "8bit", "bf16"], {"default": "4bit"}),
                "视觉Token预算": (VISUAL_TOKEN_BUDGETS, {"default": "140"}),
                "最大生成长度": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HF Mirror", "HuggingFace"], {"default": "ModelScope"}),
            },
            "optional": {
                "音频": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回复内容",)
    FUNCTION = "analyze"
    CATEGORY = "💬 AI人工智能/谷歌系列"
    DESCRIPTION = (
        "基于 Gemma 4 的视频理解节点 (隔离进程运行)。"
        "接\"加载视频\"的图像序列输出，均匀抽帧后理解视频内容，可选同时接入音频。"
        "抽帧多时建议用低视觉Token预算(70/140)。"
        "注意: 画面理解不受语言限制，但可选音频输入不支持中文语音 "
        "(官方音频基准注明 Excluding Chinese)，中文音轨请断开音频或改用 Qwen 语音识别 (ASR) 节点。"
    )

    def analyze(self, 图像序列, 任务类型, 自定义问题, 抽帧数量, 模型名称, 量化方式,
                视觉Token预算, 最大生成长度, seed, 运行后立即卸载, 自动下载模型, 下载源, 音频=None):
        prompt = VIDEO_TASK_PROMPTS.get(任务类型)
        if prompt is None:
            prompt = 自定义问题.strip()
            if not prompt:
                raise ValueError("任务类型为\"自定义问题\"时，请在\"自定义问题\"中输入内容")

        # 均匀抽帧
        total = 图像序列.shape[0]
        count = min(抽帧数量, total)
        indices = np.linspace(0, total - 1, count).astype(int)

        tmp_files = [_save_temp_image(图像序列[i]) for i in indices]
        payload = {
            "quant": 量化方式,
            "image_paths": list(tmp_files),  # 传副本，避免后续追加音频路径污染图像列表
            "prompt": prompt,
            "visual_token_budget": int(视觉Token预算),
            "max_new_tokens": 最大生成长度,
            "seed": seed,
        }
        if 音频 is not None:
            payload["audio_path"] = _save_temp_audio(音频)
            tmp_files.append(payload["audio_path"])

        try:
            content, _ = _run_gemma(模型名称, 自动下载模型, 下载源, payload, unload_after=运行后立即卸载)
            return (content,)
        finally:
            _cleanup_files(tmp_files)
