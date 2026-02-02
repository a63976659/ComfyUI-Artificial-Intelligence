import torch
import numpy as np
import random
import re
import torchaudio
from .utils import load_tts_model_data

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

def _set_seed(seed):
    if seed is not None:
        seed = seed & 0xffffffff
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

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

# ================= 节点 1: CustomVoice (预设角色) =================
class Qwen_TTS_Node:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
                
                # 改回标准名称 'seed' 以启用 ComfyUI 的自动控制组件 (Fixed/Increment/Randomize)
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
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频输出",)
    FUNCTION = "generate_speech"
    CATEGORY = "💬 AI人工智能"
    DESCRIPTION = "【预设角色模式】\n基于官方9位预设角色。支持情感指令控制。"

    def generate_speech(self, 文本内容, 模型名称, 语言, 说话人, 情感指令, seed, 温度, Top_P, Top_K, 重复惩罚, 最大生成长度, 输出模式, 下载源, 自动下载模型):
        _set_seed(seed)
        model = load_tts_model_data(模型名称, self.device, 自动下载模型, source=下载源)

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
            
            audio_segments_np = []
            sr = 24000 

            for seg_type, content in segments:
                if seg_type == "pause":
                    if content > 0: audio_segments_np.append(np.zeros(int(content * sr), dtype=np.float32))
                else:
                    instruct_text = 情感指令.strip() if 情感指令.strip() else None
                    wavs, current_sr = model.generate_custom_voice(
                        text=[content], language=[target_language], speaker=[target_speaker],
                        instruct=[instruct_text] if instruct_text else None, **gen_kwargs
                    )
                    sr = current_sr
                    if len(wavs) > 0: audio_segments_np.append(wavs[0].squeeze() if wavs[0].ndim > 1 else wavs[0])

            return _process_output(audio_segments_np, sr, 输出模式)

        except Exception as e:
            raise Exception(f"CustomVoice 生成失败: {str(e)}")

# ================= 节点 2: VoiceDesign (文本捏音) =================
class Qwen_TTS_VoiceDesign_Node:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        presets = ["Qwen3-TTS-12Hz-1.7B-VoiceDesign"]
        return {
            "required": {
                "文本内容": ("STRING", {"multiline": True, "default": "你好，这是一段测试语音。"}),
                "模型名称": (presets, {"default": presets[0]}),
                "语言": (list(LANGUAGE_MAPPING.keys()), {"default": "自动识别 (Auto)"}),
                "声音设计描述": ("STRING", {"multiline": False, "default": "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。", "placeholder": "描述声音特征、性别、年龄"}),
                
                # 改回标准名称 'seed'
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
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频输出",)
    FUNCTION = "generate_voice_design"
    CATEGORY = "💬 AI人工智能"
    DESCRIPTION = "【文本捏音模式】\n通过文字描述创造声音。"

    def generate_voice_design(self, 文本内容, 模型名称, 语言, 声音设计描述, seed, 温度, Top_P, Top_K, 重复惩罚, 最大生成长度, 输出模式, 下载源, 自动下载模型):
        _set_seed(seed)
        model = load_tts_model_data(模型名称, self.device, 自动下载模型, source=下载源)

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
            
            audio_segments_np = []
            sr = 24000 
            for seg_type, content in segments:
                if seg_type == "pause":
                    if content > 0: audio_segments_np.append(np.zeros(int(content * sr), dtype=np.float32))
                else:
                    wavs, current_sr = model.generate_voice_design(
                        text=[content], language=[target_language], 
                        instruct=[声音设计描述.strip()], **gen_kwargs
                    )
                    sr = current_sr
                    if len(wavs) > 0: audio_segments_np.append(wavs[0].squeeze() if wavs[0].ndim > 1 else wavs[0])

            return _process_output(audio_segments_np, sr, 输出模式)

        except Exception as e:
            raise Exception(f"VoiceDesign 生成失败: {str(e)}")

# ================= 节点 3: VoiceClone (语音克隆) =================
class Qwen_TTS_VoiceClone_Node:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        presets = ["Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS-12Hz-0.6B-Base"]
        return {
            "required": {
                "参考音频": ("AUDIO", ),
                "文本内容": ("STRING", {"multiline": True, "default": "通过克隆你的声音，我说出了这句话。"}),
                "模型名称": (presets, {"default": presets[0]}),
                "语言": (list(LANGUAGE_MAPPING.keys()), {"default": "自动识别 (Auto)"}),
                
                # --- 新增顺序调整：紧跟语言组件 ---
                "参考音频文本": ("STRING", {"multiline": True, "default": "", "placeholder": "(可选) 输入参考音频的文字内容。若留空则强制使用极速模式。"}),
                "情感指令": ("STRING", {"multiline": False, "default": "", "placeholder": "(可选) 例如：悲伤、开心"}),
                "极速模式": ("BOOLEAN", {"default": False, "label": "极速模式 (忽略参考文本)"}),
                
                # 改回标准名称 'seed'
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
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频输出",)
    FUNCTION = "generate_voice_clone"
    CATEGORY = "💬 AI人工智能"
    DESCRIPTION = "【语音克隆模式】\n若'参考音频文本'为空，将自动强制启用极速模式以避免报错。"

    def generate_voice_clone(self, 参考音频, 文本内容, 模型名称, 语言, seed, 温度, Top_P, Top_K, 重复惩罚, 最大生成长度, 输出模式, 下载源, 自动下载模型, 子生成器_温度, 子生成器_Top_P, 子生成器_Top_K,
                             参考音频文本="", 情感指令="", 极速模式=False):
        _set_seed(seed)
        model = load_tts_model_data(模型名称, self.device, 自动下载模型, source=下载源)

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
            
            # 1. 处理参考音频
            ref_wav_np, ref_sr = _process_ref_audio(参考音频)
            
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
                # 只有在关闭极速模式且有文本时，才传入文本
                if not final_x_vector_mode:
                    ref_text_arg = clean_ref_text

            print(f"[Qwen Clone] Extracting features... Mode={'X-Vector(Fast)' if final_x_vector_mode else 'ICL(Quality)'}")
            
            voice_prompt = model.create_voice_clone_prompt(
                ref_audio=(ref_wav_np, ref_sr),
                ref_text=ref_text_arg,
                x_vector_only_mode=final_x_vector_mode
            )

            # 3. 生成音频
            segments = _parse_text_with_pauses(文本内容)
            if not segments: raise ValueError("文本内容不能为空")
            
            audio_segments_np = []
            sr = 24000 
            for seg_type, content in segments:
                if seg_type == "pause":
                    if content > 0: audio_segments_np.append(np.zeros(int(content * sr), dtype=np.float32))
                else:
                    instruct_text = 情感指令.strip() if 情感指令 and 情感指令.strip() else None
                    wavs, current_sr = model.generate_voice_clone(
                        text=[content],
                        language=[target_language],
                        voice_clone_prompt=voice_prompt,
                        instruct=[instruct_text] if instruct_text else None,
                        **gen_kwargs
                    )
                    sr = current_sr
                    if len(wavs) > 0: audio_segments_np.append(wavs[0].squeeze() if wavs[0].ndim > 1 else wavs[0])

            return _process_output(audio_segments_np, sr, 输出模式)

        except Exception as e:
            raise Exception(f"VoiceClone 生成失败: {str(e)}")