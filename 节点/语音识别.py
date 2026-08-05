# -*- coding: utf-8 -*-
"""
Qwen3-ASR 语音识别节点 (隔离环境子进程运行)
模型统一存放于 models/TTS 下，时间戳功能需额外的 Qwen3-ForcedAligner-0.6B 模型。
"""
import os
import json
import tempfile

import soundfile as sf
from comfy.utils import ProgressBar

from .utils import resolve_tts_model
from .隔离环境 import run_worker

# ================= 配置与常量 =================

ASR_MODELS = ["Qwen3-ASR-1.7B", "Qwen3-ASR-0.6B"]
ALIGNER_MODEL = "Qwen3-ForcedAligner-0.6B"

ASR_LANGUAGES = {
    "自动识别 (Auto)": None,
    "中文 (Chinese)": "Chinese",
    "英语 (English)": "English",
    "粤语 (Cantonese)": "Cantonese",
    "日语 (Japanese)": "Japanese",
    "韩语 (Korean)": "Korean",
    "法语 (French)": "French",
    "德语 (German)": "German",
    "西班牙语 (Spanish)": "Spanish",
    "俄语 (Russian)": "Russian",
    "意大利语 (Italian)": "Italian",
    "葡萄牙语 (Portuguese)": "Portuguese",
    "泰语 (Thai)": "Thai",
    "越南语 (Vietnamese)": "Vietnamese",
    "阿拉伯语 (Arabic)": "Arabic",
    "印尼语 (Indonesian)": "Indonesian",
    "土耳其语 (Turkish)": "Turkish",
    "印地语 (Hindi)": "Hindi",
    "马来语 (Malay)": "Malay",
    "荷兰语 (Dutch)": "Dutch",
    "瑞典语 (Swedish)": "Swedish",
    "丹麦语 (Danish)": "Danish",
    "芬兰语 (Finnish)": "Finnish",
    "波兰语 (Polish)": "Polish",
    "捷克语 (Czech)": "Czech",
    "菲律宾语 (Filipino)": "Filipino",
    "波斯语 (Persian)": "Persian",
    "希腊语 (Greek)": "Greek",
    "匈牙利语 (Hungarian)": "Hungarian",
    "马其顿语 (Macedonian)": "Macedonian",
    "罗马尼亚语 (Romanian)": "Romanian",
}

# ================= 辅助函数 =================

def _save_audio_to_temp(audio_input):
    """将 ComfyUI 的音频 Tensor 存为临时 WAV 文件，供子进程读取

    使用 soundfile 写出，避开 torchaudio 依赖 torchcodec / FFmpeg 的问题 (与加载音频节点一致)。
    """
    waveform = audio_input["waveform"]
    sample_rate = audio_input["sample_rate"]

    wav_tensor = waveform[0] if waveform.dim() == 3 else waveform
    if wav_tensor.shape[0] > wav_tensor.shape[1]:
        wav_tensor = wav_tensor.t()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.close()
    # soundfile 需要 (frames, channels)，而 ComfyUI 音频为 (channels, frames)
    sf.write(temp_file.name, wav_tensor.t().cpu().numpy(), int(sample_rate))
    return temp_file.name


def _format_time(seconds):
    """将秒数格式化为 分:秒.毫秒 (MM:SS.xx)"""
    if seconds is None:
        return "00:00.00"
    seconds = float(seconds)
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def _format_timestamps(merged_stamps):
    """将时间戳字典列表格式化为逐行文本"""
    if not merged_stamps:
        return ""
    return "\n".join(
        f"{_format_time(ts.get('start', 0.0))} - {_format_time(ts.get('end', 0.0))}: {ts.get('text', '')}"
        for ts in merged_stamps
    )


def _resolve_models(模型名称, 生成时间戳, 自动下载模型, 下载源):
    """定位/下载主模型与 (可选的) 对齐模型，返回两个本地路径"""
    model_path = resolve_tts_model(模型名称, 自动下载模型, source=下载源)

    aligner_path = None
    if 生成时间戳:
        try:
            aligner_path = resolve_tts_model(ALIGNER_MODEL, 自动下载模型, source=下载源)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"生成时间戳需要额外的对齐模型 {ALIGNER_MODEL}，本地未找到，"
                "请开启\"自动下载模型\"或手动下载到 models/TTS 目录"
            )
    return model_path, aligner_path


def _transcribe_one(audio_item, model_path, aligner_path, language, context, 生成时间戳, unload_after):
    """单条音频识别: 存临时 wav -> 隔离环境子进程识别 -> 返回响应"""
    temp_path = _save_audio_to_temp(audio_item)
    try:
        return run_worker(
            "asr",
            {
                "model_path": model_path,
                "aligner_path": aligner_path,
                "audio_path": temp_path,
                "language": language,
                "context": context,
                "return_time_stamps": 生成时间戳,
            },
            unload_after=unload_after,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ================= 节点 1: 标准语音识别 =================

class Qwen语音识别_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO",),
                "模型名称": (ASR_MODELS, {"default": ASR_MODELS[0]}),
                "语言": (list(ASR_LANGUAGES.keys()), {"default": "自动识别 (Auto)"}),
                "提示词": ("STRING", {"multiline": True, "default": "", "placeholder": "可选：输入上下文或提示词"}),
                "生成时间戳": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HuggingFace", "HF Mirror"], {"default": "ModelScope"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("文本输出", "JSON详细数据", "带时间戳文本")
    OUTPUT_NODE = True
    FUNCTION = "transcribe_audio"
    CATEGORY = "💬 AI人工智能/千问系列"
    DESCRIPTION = (
        "使用 Qwen3-ASR 进行语音识别 (隔离进程运行，不影响其它节点)。\n"
        "开启\"生成时间戳\"需额外下载 Qwen3-ForcedAligner-0.6B 对齐模型。"
    )

    def transcribe_audio(self, 音频, 模型名称, 语言, 提示词, 生成时间戳, 下载源, 自动下载模型,
                         运行后立即卸载=True):
        model_path, aligner_path = _resolve_models(模型名称, 生成时间戳, 自动下载模型, 下载源)

        resp = _transcribe_one(
            音频, model_path, aligner_path,
            ASR_LANGUAGES.get(语言),
            提示词.strip() or None,
            生成时间戳,
            unload_after=运行后立即卸载,
        )

        text_output = resp.get("text", "")
        timestamps = resp.get("timestamps", [])
        json_data = {
            "language": resp.get("language"),
            "text": text_output,
            "timestamps": timestamps,
        }
        return (
            text_output,
            json.dumps(json_data, ensure_ascii=False, indent=2),
            _format_timestamps(timestamps),
        )


# ================= 节点 2: 批量语音识别 =================

class Qwen批量语音识别_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频列表": ("AUDIO",),
                "模型名称": (ASR_MODELS, {"default": ASR_MODELS[0]}),
                "语言": (list(ASR_LANGUAGES.keys()), {"default": "自动识别 (Auto)"}),
                "提示词": ("STRING", {"multiline": True, "default": "", "placeholder": "批量提示词"}),
                "生成时间戳": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HuggingFace", "HF Mirror"], {"default": "ModelScope"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("合并文本", "详细日志文本", "带时间戳文本")
    OUTPUT_NODE = True
    FUNCTION = "batch_transcribe"
    CATEGORY = "💬 AI人工智能/千问系列"
    DESCRIPTION = (
        "批量处理多个音频片段 (隔离进程运行)，输出合并后的文本。\n"
        "可直接连接\"批量加载音频\"节点，模型在整批处理期间只加载一次。"
    )

    def batch_transcribe(self, 音频列表, 模型名称, 语言, 提示词, 生成时间戳, 下载源, 自动下载模型,
                         运行后立即卸载=True):
        audio_inputs = 音频列表 if isinstance(音频列表, list) else [音频列表]
        total_files = len(audio_inputs)
        if total_files == 0:
            return ("", "", "")

        model_path, aligner_path = _resolve_models(模型名称, 生成时间戳, 自动下载模型, 下载源)
        target_lang = ASR_LANGUAGES.get(语言)
        context_prompt = 提示词.strip() or None

        pbar = ProgressBar(total_files)
        full_text_list = []
        log_lines = []
        timestamp_text_list = []

        print(f"[Qwen语音识别] 批量处理 {total_files} 个音频...")

        for i, audio_item in enumerate(audio_inputs):
            # 仅最后一条按用户设置卸载，中间保持常驻以避免反复加载模型
            is_last = (i == total_files - 1)
            filename = audio_item.get("filename", f"Audio_{i + 1}") if isinstance(audio_item, dict) else f"Audio_{i + 1}"
            try:
                resp = _transcribe_one(
                    audio_item, model_path, aligner_path, target_lang, context_prompt, 生成时间戳,
                    unload_after=(运行后立即卸载 and is_last),
                )
                text = resp.get("text", "")
                full_text_list.append(text)

                current_ts_str = _format_timestamps(resp.get("timestamps", []))
                if current_ts_str:
                    timestamp_text_list.append(f"--- {filename} ---")
                    timestamp_text_list.append(current_ts_str)
                    timestamp_text_list.append("")

                log_lines.append(f"--- [{i + 1}/{total_files}] {filename} ({resp.get('language')}) ---")
                log_lines.append(text)
                if current_ts_str:
                    log_lines.append("[Timestamps]")
                    log_lines.append(current_ts_str)
                log_lines.append("")
            except Exception as inner_e:
                print(f"[Qwen语音识别] 第 {i + 1} 个音频处理失败: {inner_e}")
                full_text_list.append(f"[Error in file {i + 1}]")
                log_lines.append(f"--- [{i + 1}/{total_files}] {filename} (失败) ---")
                log_lines.append(str(inner_e))
                log_lines.append("")
            finally:
                pbar.update(1)

        return ("\n".join(full_text_list), "\n".join(log_lines), "\n".join(timestamp_text_list))
