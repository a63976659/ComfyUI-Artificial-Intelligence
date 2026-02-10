import os
import torchaudio
import folder_paths
import hashlib

# ================= 节点 1: 批量加载音频 (保持不变) =================

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
                waveform, sample_rate = torchaudio.load(file_path)
                batch_audio_data.append({
                    "waveform": waveform.unsqueeze(0) if waveform.dim() == 2 else waveform,
                    "sample_rate": sample_rate,
                    "filename": os.path.basename(file_path),
                })
            except Exception as e:
                print(f"[Error] Failed to load {file_path}: {e}")

        return (batch_audio_data, len(batch_audio_data))

# ================= 节点 2: 单个加载音频 (UI增强版) =================

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
    DESCRIPTION = "加载单个音频，支持自动获取时长、波形预览和裁剪。"

    def load_audio(self, 文件路径, 开始时间, 持续时间):
        # 处理路径：优先尝试作为绝对路径，如果不存在则尝试在 input 目录下查找
        path = 文件路径.strip().strip('"')
        
        if not os.path.isabs(path):
            # 尝试在 ComfyUI 的 input 目录查找
            possible_path = folder_paths.get_annotated_filepath(path)
            if possible_path:
                path = possible_path
            else:
                path = os.path.abspath(path)
            
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            info = torchaudio.info(path)
            sr = info.sample_rate
            total_frames = info.num_frames
            
            frame_offset = int(开始时间 * sr)
            # 如果持续时间为0，则读取到最后；否则读取指定长度
            num_frames = int(持续时间 * sr) if 持续时间 > 0 else -1
            
            if frame_offset >= total_frames:
                frame_offset = 0
            
            waveform, sample_rate = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
            
            # 保存预览 (temp)
            params_hash = hashlib.md5(f"{path}_{开始时间}_{持续时间}".encode("utf-8")).hexdigest()
            preview_filename = f"preview_{params_hash}.wav"
            preview_dir = folder_paths.get_temp_directory()
            preview_path = os.path.join(preview_dir, preview_filename)
            
            torchaudio.save(preview_path, waveform, sample_rate)

            # 返回 ui 数据给 JS 使用 (temp 类型)
            return {
                "ui": {
                    "audio": [{
                        "filename": preview_filename,
                        "subfolder": "",
                        "type": "temp"
                    }]
                },
                "result": ({
                    "waveform": waveform.unsqueeze(0) if waveform.dim() == 2 else waveform, 
                    "sample_rate": sample_rate,
                    "filename": os.path.basename(path),
                    "path": path
                },)
            }

        except Exception as e:
            print(f"[LoadAudio Error] {e}")
            raise Exception(f"Failed to load audio: {str(e)}")

    @classmethod
    def IS_CHANGED(s, 文件路径, 开始时间, 持续时间):
        return hashlib.md5(f"{文件路径}_{开始时间}_{持续时间}".encode("utf-8")).hexdigest()

