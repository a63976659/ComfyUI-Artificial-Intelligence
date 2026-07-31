import os
import re
import uuid
import shutil
import torch
import numpy as np
import torchaudio
import folder_paths
from .utils import resolve_tts_model
from .隔离环境 import run_worker

# ================= 映射字典 =================
SPEAKER_MAPPING = {
    "Vivian (中文-明亮微急)": "Vivian",
    "Serena (中文-温暖温柔)": "Serena",
    "Uncle_Fu (中文-醇厚男声)": "Uncle_Fu",
    "Dylan (中文-北京少年)": "Dylan",
    "Eric (中文-四川话)": "Eric",
    "Ryan (英文-动感节奏)": "Ryan",
    "Aiden (英文-阳光男声)": "Aiden",
    "Ono_Anna (日文-俏皮灵动)": "Ono_Anna",
    "Sohee (韩文-温暖情感)": "Sohee"
}

LANGUAGE_MAPPING = {
    "自动识别 (Auto)": "Auto",
    "中文 (Chinese)": "Chinese",
    "英文 (English)": "English",
    "日文 (Japanese)": "Japanese",
    "韩文 (Korean)": "Korean",
    "德文 (German)": "German",
    "法文 (French)": "French",
    "俄文 (Russian)": "Russian",
    "葡萄牙文 (Portuguese)": "Portuguese",
    "西班牙文 (Spanish)": "Spanish",
    "意大利文 (Italian)": "Italian"
}

# ================= 通用辅助函数 =================
def _parse_text_with_pauses(text_input):
    input_lines = [t.strip() for t in text_input.split("\n") if t.strip()]
    segments = []
    pause_pattern = re.compile(r"\[(?:pause|p):(\d+(?:\.\d+)?)\]", re.IGNORECASE)

    for line in input_lines:
        last_idx = 0
        for match in pause_pattern.finditer(line):
            text_part = line[last_idx : match.start()].strip()
            if text_part:
                segments.append(("text", text_part))
            try:
                duration = float(match.group(1))
                segments.append(("pause", duration))
            except ValueError:
                pass
            last_idx = match.end()

        remaining_text = line[last_idx:].strip()
        if remaining_text:
            segments.append(("text", remaining_text))
    return segments

def _process_ref_audio(audio_dict):
    """处理参考音频：转换为单声道并重采样到 16k"""
    waveform = audio_dict['waveform'] 
    sr = audio_dict['sample_rate']
    
    if waveform.dim() == 3:
        waveform = waveform[0] 
    
    if waveform.shape[0] > waveform.shape[1] and waveform.shape[0] > 100:
        waveform = waveform.t()

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    wav_numpy = waveform.squeeze().cpu().numpy()
    return wav_numpy, target_sr

def _process_output(audio_segments_np, sr, output_mode):
    """统一处理输出模式：拼合 或 批次"""
    if not audio_segments_np:
        raise Exception("未生成音频")

    if output_mode == "拼合 (Concatenate)":
        full_audio = np.concatenate(audio_segments_np)
        audio_tensor = torch.from_numpy(full_audio).float()
        # (1, 1, Samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        return ({"waveform": audio_tensor, "sample_rate": sr},)
    
    else: # 批次 (Batch)
        max_len = max(len(seg) for seg in audio_segments_np)
        batch_size = len(audio_segments_np)
        batch_tensor = torch.zeros(batch_size, 1, max_len, dtype=torch.float32)
        
        for i, seg in enumerate(audio_segments_np):
            tensor_seg = torch.from_numpy(seg).float()
            length = tensor_seg.shape[0]
            batch_tensor[i, 0, :length] = tensor_seg
            
        return ({"waveform": batch_tensor, "sample_rate": sr},)

def _run_tts(模型名称, 自动下载模型, 下载源, payload, unload_after):
    """公共流程: 定位/下载模型 -> 隔离环境子进程合成 -> 读回波形段 (.npy)"""
    out_dir = os.path.join(folder_paths.get_temp_directory(), f"tts_{uuid.uuid4().hex}")
    os.makedirs(out_dir, exist_ok=True)
    payload["output_dir"] = out_dir
    payload["model_path"] = resolve_tts_model(模型名称, 自动下载模型, source=下载源)
    try:
        resp = run_worker("tts", payload, unload_after=unload_after)
        audio_segments_np = [np.load(p) for p in resp.get("segment_paths", [])]
        return audio_segments_np, int(resp.get("sample_rate", 24000))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

# ================= 节点 1: CustomVoice (预设角色) =================
class Qwen_TTS_Node:
    @classmethod
    def INPUT_TYPES(cls):
        presets = ["Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3-TTS-12Hz-0.6B-CustomVoice"]
        return {
            "required": {
                "文本内容": ("STRING", {"multiline": True, "default": "你好，我是Vivian。"}),
                "模型名称": (presets, {"default": presets[0]}),
                "语言": (list(LANGUAGE_MAPPING.keys()), {"default": "自动识别 (Auto)"}),
                "说话人": (list(SPEAKER_MAPPING.keys()), {"default": "Vivian (中文-明亮微急)"}),
                "情感指令": ("STRING", {"multiline": False, "default": "高兴", "placeholder": "例如：高兴、悲伤"}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                
                # --- 主生成参数 ---
                "温度": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
                "Top_P": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "Top_K": ("INT", {"default": 50, "min": 0, "max": 100}),
                "重复惩罚": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 2.0, "step": 0.05}),
                "最大生成长度": ("INT", {"default": 2048, "min": 128, "max": 8192}),
                
                "输出模式": (["拼合 (Concatenate)", "批次 (Batch)"], {"default": "拼合 (Concatenate)"}),
                
                # --- 下载相关 (放在最后) ---
                "下载源": (["ModelScope", "HuggingFace", "HF Mirror"], {"default": "ModelScope"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频输出",)
    FUNCTION = "generate_speech"
    CATEGORY = "💬 AI人工智能/千问系列"
    DESCRIPTION = (
        "【Qwen CustomVoice - 预设角色模式】\n"
        "✨ 使用介绍：\n"
        "1. 文本内容：输入要合成的文字，支持多行，支持 [pause:0.5] 插入0.5秒停顿。\n"
        "2. 模型名称：选择 1.7B (高质量) 或 0.6B (速度快) 模型。\n"
        "3. 语言：推荐 Auto，支持中英日韩等多语言混合。\n"
        "4. 说话人：选择 9 位官方预设的高质量角色。\n"
        "5. 情感指令：输入形容词控制语气，如“悲伤”、“激昂”、“窃窃私语”。\n"
        "6. seed：随机种子，固定后可复现相同的语音效果。\n"
        "7. 生成参数(温度/Top_P等)：控制生成的随机性和多样性。\n"
        "8. 输出模式：'拼合'将所有文本合成一条音频；'批次'将每行文本单独输出。\n"
        "⚠️ 注意：小显卡可以选择0.6B模型。"
    )

    def generate_speech(self, 文本内容, 模型名称, 语言, 说话人, 情感指令, seed, 温度, Top_P, Top_K, 重复惩罚, 最大生成长度, 输出模式, 下载源, 自动下载模型, 运行后立即卸载=True):
        try:
            target_speaker = SPEAKER_MAPPING.get(说话人, "Vivian")
            target_language = LANGUAGE_MAPPING.get(语言, "Auto")
            gen_kwargs = {
                "temperature": 温度, "top_p": Top_P, "top_k": Top_K,
                "repetition_penalty": 重复惩罚, "max_new_tokens": 最大生成长度,
                "subtalker_dosample": True 
            }
            
            segments = _parse_text_with_pauses(文本内容)
            if not segments: raise ValueError("文本内容不能为空")

            instruct_text = 情感指令.strip() if 情感指令.strip() else None
            audio_segments_np, sr = _run_tts(
                模型名称, 自动下载模型, 下载源,
                {
                    "mode": "custom_voice",
                    "segments": segments,
                    "language": target_language,
                    "speaker": target_speaker,
                    "instruct": instruct_text,
                    "gen_kwargs": gen_kwargs,
                    "seed": seed,
                },
                unload_after=运行后立即卸载,
            )
            return _process_output(audio_segments_np, sr, 输出模式)

        except Exception as e:
            raise Exception(f"CustomVoice 生成失败: {str(e)}")

# ================= 节点 2: VoiceDesign (文本捏音) =================
class Qwen_TTS_VoiceDesign_Node:
    @classmethod
    def INPUT_TYPES(cls):
        presets = ["Qwen3-TTS-12Hz-1.7B-VoiceDesign"]
        return {
            "required": {
                "文本内容": ("STRING", {"multiline": True, "default": "你好，这是一段测试语音。"}),
                "模型名称": (presets, {"default": presets[0]}),
                "语言": (list(LANGUAGE_MAPPING.keys()), {"default": "自动识别 (Auto)"}),
                "声音设计描述": ("STRING", {"multiline": False, "default": "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。", "placeholder": "描述声音特征、性别、年龄"}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                
                # --- 主生成参数 ---
                "温度": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
                "Top_P": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "Top_K": ("INT", {"default": 50, "min": 0, "max": 100}),
                "重复惩罚": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 2.0, "step": 0.05}),
                "最大生成长度": ("INT", {"default": 2048, "min": 128, "max": 8192}),
                
                "输出模式": (["拼合 (Concatenate)", "批次 (Batch)"], {"default": "拼合 (Concatenate)"}),
                
                # --- 下载相关 (放在最后) ---
                "下载源": (["ModelScope", "HuggingFace", "HF Mirror"], {"default": "ModelScope"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频输出",)
    FUNCTION = "generate_voice_design"
    CATEGORY = "💬 AI人工智能/千问系列"
    DESCRIPTION = (
        "【Qwen VoiceDesign - 文本捏音模式】\n"
        "✨ 使用介绍：\n"
        "1. 文本内容：输入要合成的文字。\n"
        "2. 声音设计描述：[核心] 用自然语言描述你想要的声音。例如：“一个中年男性，嗓音沙哑，语气严肃”、“年轻活泼的女孩，声音甜美”。\n"
        "3. 模型名称：仅支持 1.7B VoiceDesign 模型。\n"
        "4. seed：不同的种子会产生略微不同的音色细节。\n"
        "5. 生成参数：调节温度等可改变声音的变化幅度。\n"
        "⚠️ 注意：第一次使用可以开启自动下载模型，默认下载源适合国内网络。"
    )

    def generate_voice_design(self, 文本内容, 模型名称, 语言, 声音设计描述, seed, 温度, Top_P, Top_K, 重复惩罚, 最大生成长度, 输出模式, 下载源, 自动下载模型, 运行后立即卸载=True):
        try:
            target_language = LANGUAGE_MAPPING.get(语言, "Auto")
            gen_kwargs = {
                "temperature": 温度, "top_p": Top_P, "top_k": Top_K,
                "repetition_penalty": 重复惩罚, "max_new_tokens": 最大生成长度,
                "subtalker_dosample": True
            }
            segments = _parse_text_with_pauses(文本内容)
            if not segments: raise ValueError("文本内容不能为空")
            if not 声音设计描述.strip(): raise ValueError("声音设计描述不能为空")

            audio_segments_np, sr = _run_tts(
                模型名称, 自动下载模型, 下载源,
                {
                    "mode": "voice_design",
                    "segments": segments,
                    "language": target_language,
                    "instruct": 声音设计描述.strip(),
                    "gen_kwargs": gen_kwargs,
                    "seed": seed,
                },
                unload_after=运行后立即卸载,
            )
            return _process_output(audio_segments_np, sr, 输出模式)

        except Exception as e:
            raise Exception(f"VoiceDesign 生成失败: {str(e)}")

# ================= 节点 3: VoiceClone (语音克隆) =================
class Qwen_TTS_VoiceClone_Node:
    @classmethod
    def INPUT_TYPES(cls):
        presets = ["Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS-12Hz-0.6B-Base"]
        return {
            "required": {
                "参考音频": ("AUDIO", ),
                "文本内容": ("STRING", {"multiline": True, "default": "通过克隆你的声音，我说出了这句话。"}),
                "模型名称": (presets, {"default": presets[0]}),
                "语言": (list(LANGUAGE_MAPPING.keys()), {"default": "自动识别 (Auto)"}),
                
                # --- 组件改为单行 ---
                "参考音频文本": ("STRING", {"multiline": False, "default": "", "placeholder": "(可选) 输入参考音频的文字内容。若留空则强制使用极速模式。"}),
                "情感指令": ("STRING", {"multiline": False, "default": "生气", "placeholder": "(可选) 例如：悲伤、开心"}),
                "极速模式": ("BOOLEAN", {"default": False, "label": "极速模式 (忽略参考文本)"}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                
                # --- 主生成参数 ---
                "温度": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
                "Top_P": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "Top_K": ("INT", {"default": 50, "min": 0, "max": 100}),
                "重复惩罚": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 2.0, "step": 0.05}),
                "最大生成长度": ("INT", {"default": 2048, "min": 128, "max": 8192}),
                
                "输出模式": (["拼合 (Concatenate)", "批次 (Batch)"], {"default": "拼合 (Concatenate)"}),
                
                # --- 子生成器参数 (仅克隆节点保留) ---
                "子生成器_温度": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.05}),
                "子生成器_Top_P": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "子生成器_Top_K": ("INT", {"default": 50, "min": 0, "max": 100}),
                
                # --- 下载相关 (放在最后) ---
                "下载源": (["ModelScope", "HuggingFace", "HF Mirror"], {"default": "ModelScope"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频输出",)
    FUNCTION = "generate_voice_clone"
    CATEGORY = "💬 AI人工智能/千问系列"
    DESCRIPTION = (
        "【Qwen VoiceClone - 语音克隆模式】\n"
        "✨ 使用介绍：\n"
        "1. 参考音频：输入一段 5-10秒 清晰的人声录音。\n"
        "2. 参考音频文本：[推荐] 输入这段音频里说的话。这能大幅提高克隆相似度（ICL模式）。若留空，则自动切换为“极速模式”。\n"
        "3. 极速模式：勾选后将忽略参考文本，仅提取音色特征（X-Vector）。速度快，但相似度略低。\n"
        "4. 情感指令：即使是克隆声音，也可以要求它用“悲伤”或“开心”的语气说话。\n"
        "5. 子生成器参数：克隆模式特有参数，用于微调声学细节的随机性，建议保持默认。\n"
        "6. 文本内容：希望克隆声音说出的新内容。\n"
        "⚠️ 注意：必须连接 参考音频 节点使用。"
    )

    def generate_voice_clone(self, 参考音频, 文本内容, 模型名称, 语言, seed, 温度, Top_P, Top_K, 重复惩罚, 最大生成长度, 输出模式, 下载源, 自动下载模型, 子生成器_温度, 子生成器_Top_P, 子生成器_Top_K,
                             参考音频文本="", 情感指令="", 极速模式=False, 运行后立即卸载=True):
        ref_audio_path = None
        try:
            target_language = LANGUAGE_MAPPING.get(语言, "Auto")
            gen_kwargs = {
                "temperature": 温度, "top_p": Top_P, "top_k": Top_K,
                "repetition_penalty": 重复惩罚, "max_new_tokens": 最大生成长度,
                "subtalker_temperature": 子生成器_温度,
                "subtalker_top_p": 子生成器_Top_P,
                "subtalker_top_k": 子生成器_Top_K,
                "subtalker_dosample": True
            }

            segments = _parse_text_with_pauses(文本内容)
            if not segments: raise ValueError("文本内容不能为空")
            
            # 1. 处理参考音频 (重采样到 16k 后存为 .npy 传给子进程)
            ref_wav_np, ref_sr = _process_ref_audio(参考音频)
            ref_audio_path = os.path.join(folder_paths.get_temp_directory(), f"tts_ref_{uuid.uuid4().hex}.npy")
            os.makedirs(os.path.dirname(ref_audio_path), exist_ok=True)
            np.save(ref_audio_path, ref_wav_np.astype(np.float32))
            
            # 2. 逻辑修正：如果参考文本为空，强制使用极速模式
            clean_ref_text = 参考音频文本.strip()
            
            # 决定最终模式
            final_x_vector_mode = 极速模式
            ref_text_arg = None
            
            if not clean_ref_text:
                if not final_x_vector_mode:
                    print("[Qwen TTS Warning] 未填写参考音频文本，已自动切换至 '极速模式' (X-Vector Only)。")
                    final_x_vector_mode = True
                ref_text_arg = None
            else:
                if not final_x_vector_mode:
                    ref_text_arg = clean_ref_text

            # 3. 隔离环境子进程合成
            instruct_text = 情感指令.strip() if 情感指令 and 情感指令.strip() else None
            audio_segments_np, sr = _run_tts(
                模型名称, 自动下载模型, 下载源,
                {
                    "mode": "voice_clone",
                    "segments": segments,
                    "language": target_language,
                    "instruct": instruct_text,
                    "gen_kwargs": gen_kwargs,
                    "seed": seed,
                    "ref_audio_path": ref_audio_path,
                    "ref_sample_rate": ref_sr,
                    "ref_text": ref_text_arg,
                    "x_vector_only": final_x_vector_mode,
                },
                unload_after=运行后立即卸载,
            )
            return _process_output(audio_segments_np, sr, 输出模式)

        except Exception as e:
            raise Exception(f"VoiceClone 生成失败: {str(e)}")
        finally:
            if ref_audio_path:
                try:
                    os.remove(ref_audio_path)
                except OSError:
                    pass