import os
import shutil
import soundfile as sf
import folder_paths
import hashlib
import subprocess
import torch

# ================= 通用: ffmpeg 定位 =================
_FFMPEG_EXE = None

def _get_ffmpeg_exe():
    """定位 ffmpeg 可执行文件: 优先系统 PATH (秋叶整合包自带)，
    官方便携版/桌面版无 ffmpeg 时回退 imageio-ffmpeg 内置的可执行文件。"""
    global _FFMPEG_EXE
    if _FFMPEG_EXE is None:
        exe = shutil.which("ffmpeg")
        if not exe:
            try:
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                raise RuntimeError(
                    "未找到 ffmpeg: 请安装 ffmpeg 并加入系统 PATH，"
                    "或在 ComfyUI 主环境执行 pip install imageio-ffmpeg"
                )
        _FFMPEG_EXE = exe
    return _FFMPEG_EXE

# ================= 通用: 音频读取 (不依赖 torchcodec) =================
def _load_audio_as_tensor(path):
    """使用 soundfile 读取音频，避免 torchaudio 的 torchcodec 后端依赖。
    返回 (waveform, sample_rate)，waveform 形状为 (channels, frames)，与 torchaudio.load 保持一致。
    soundfile 不支持的格式(如 m4a)会自动用 ffmpeg 转码为 wav 后再读取。"""
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        temp_dir = folder_paths.get_temp_directory()
        params_hash = hashlib.md5(path.encode("utf-8")).hexdigest()
        temp_wav = os.path.join(temp_dir, f"sf_decode_{params_hash}.wav")
        if not os.path.exists(temp_wav):
            cmd = [_get_ffmpeg_exe(), "-y", "-i", path, "-vn", "-acodec", "pcm_s16le", temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        data, sample_rate = sf.read(temp_wav, dtype="float32", always_2d=True)
    # sf.read(always_2d=True) 返回 (frames, channels)，转置为 (channels, frames)
    waveform = torch.from_numpy(data.T).contiguous()
    return waveform, sample_rate

# ================= 节点 1: 批量加载音频 =================

class 批量加载音频_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "文件夹路径": ("STRING", {"default": "./input/audio", "multiline": False, "label": "文件夹路径"}),
            },
            "optional": {
                "文件扩展名": ("STRING", {"default": "wav,mp3,flac,m4a,ogg", "multiline": False, "label": "文件扩展名"}),
                "递归搜索": ("BOOLEAN", {"default": False, "label": "递归搜索子文件夹"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("音频列表", "文件数量")
    FUNCTION = "load_batch_audio"
    CATEGORY = "💬 AI人工智能/加载音频"
    DESCRIPTION = "从指定文件夹批量加载音频文件，支持递归搜索。"

    def load_batch_audio(self, 文件夹路径, 文件扩展名, 递归搜索):
        path = 文件夹路径.strip().strip('"')
        if not os.path.isabs(path): path = os.path.abspath(path)
        
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except:
                pass
                
        if not os.path.isdir(path): return ([], 0)
        
        extensions = tuple([f".{ext.strip().lower()}" for ext in 文件扩展名.split(",")])
        audio_files = []
        
        if 递归搜索:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(extensions):
                        audio_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path) and file.lower().endswith(extensions):
                    audio_files.append(file_path)

        audio_files.sort()
        if not audio_files: return ([], 0)

        batch_audio_data = []
        for file_path in audio_files:
            try:
                waveform, sample_rate = _load_audio_as_tensor(file_path)
                batch_audio_data.append({
                    "waveform": waveform.unsqueeze(0) if waveform.dim() == 2 else waveform,
                    "sample_rate": sample_rate,
                    "filename": os.path.basename(file_path),
                })
            except Exception as e:
                print(f"[Error] Failed to load {file_path}: {e}")

        return (batch_audio_data, len(batch_audio_data))

# ================= 节点 2: 单个加载音频 =================

class 加载音频_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "文件路径": ("STRING", {"default": "example.wav", "multiline": False, "label": "文件路径"}),
            },
            "optional": {
                "开始时间": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01, "label": "开始时间(秒)"}),
                "持续时间": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01, "label": "持续时间(0=全长)"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    OUTPUT_NODE = True
    FUNCTION = "load_audio"
    CATEGORY = "💬 AI人工智能/加载音频"
    DESCRIPTION = "加载音频或视频文件，自动提取音轨，支持波形预览和裁剪。"

    def load_audio(self, 文件路径, 开始时间, 持续时间):
        path = 文件路径.strip().strip('"')
        
        if not os.path.isabs(path):
            possible_path = folder_paths.get_annotated_filepath(path)
            if possible_path:
                path = possible_path
            else:
                path = os.path.abspath(path)
            
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到音频或视频文件: {path}")

        try:
            temp_dir = folder_paths.get_temp_directory()
            params_hash = hashlib.md5(f"{path}_{开始时间}_{持续时间}".encode("utf-8")).hexdigest()
            temp_wav = os.path.join(temp_dir, f"audio_extract_{params_hash}.wav")
            
            if not os.path.exists(temp_wav):
                cmd = [_get_ffmpeg_exe(), "-y"]
                if 开始时间 > 0:
                    cmd.extend(["-ss", str(开始时间)])
                cmd.extend(["-i", path])
                if 持续时间 > 0:
                    cmd.extend(["-t", str(持续时间)])
                cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", temp_wav])
                
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if os.path.exists(temp_wav):
                waveform, sample_rate = _load_audio_as_tensor(temp_wav)
                if waveform.dim() == 2:
                    waveform = waveform.unsqueeze(0)
            else:
                waveform_full, sample_rate = _load_audio_as_tensor(path)
                total_frames = waveform_full.shape[1]
                start_frame = int(开始时间 * sample_rate)
                if start_frame >= total_frames: start_frame = 0
                if 持续时间 > 0:
                    end_frame = start_frame + int(持续时间 * sample_rate)
                    if end_frame > total_frames: end_frame = total_frames
                    waveform = waveform_full[:, start_frame:end_frame]
                else:
                    waveform = waveform_full[:, start_frame:]
                if waveform.dim() == 2:
                    waveform = waveform.unsqueeze(0)

            return ({"waveform": waveform, "sample_rate": sample_rate},)

        except Exception as e:
            print(f"[LoadAudio Error] {e}")
            raise Exception(f"Failed to load audio/video track: {str(e)}")

    @classmethod
    def IS_CHANGED(s, 文件路径, 开始时间, 持续时间):
        return hashlib.md5(f"{文件路径}_{开始时间}_{持续时间}".encode("utf-8")).hexdigest()
