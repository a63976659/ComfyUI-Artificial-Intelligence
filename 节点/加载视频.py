import os
import cv2
import torch
import numpy as np
import torchaudio
import folder_paths
import hashlib
import subprocess

def extract_video_frames(path, start_time, end_time, single_frame=False):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_time * fps)
    if single_frame:
        end_frame = start_frame
    else:
        end_frame = int(end_time * fps) if end_time > 0 else total_frames
        
    if start_frame >= total_frames:
        start_frame = total_frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    curr = start_frame
    
    while curr <= end_frame and curr < total_frames:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        curr += 1
        if single_frame: break

    cap.release()
    
    if not frames:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
    frames_np = np.array(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(frames_np)

def extract_audio(path, start_time, end_time):
    try:
        temp_dir = folder_paths.get_temp_directory()
        params_hash = hashlib.md5(f"{path}_{start_time}_{end_time}".encode("utf-8")).hexdigest()
        temp_wav = os.path.join(temp_dir, f"audio_{params_hash}.wav")
        
        if not os.path.exists(temp_wav):
            cmd = ["ffmpeg", "-y"]
            if start_time > 0:
                cmd.extend(["-ss", str(start_time)])
            cmd.extend(["-i", path])
            if end_time > 0 and end_time > start_time:
                cmd.extend(["-t", str(end_time - start_time)])
            cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", temp_wav])
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        if os.path.exists(temp_wav):
            waveform, sample_rate = torchaudio.load(temp_wav)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0) 
            return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as e:
        print(f"[视频音频提取] 失败: {e}")
        
    print(f"[视频音频提取] 采用静音占位符兜底")
    return {"waveform": torch.zeros((1, 2, 1024), dtype=torch.float32), "sample_rate": 44100}

# ================= 视频对象包装器 (Duck Typing) =================
class QwenVideoOutput(str):
    def get_dimensions(self):
        cap = cv2.VideoCapture(str(self))
        if not cap.isOpened():
            return (0, 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (w, h)
        
    def get_stream_source(self):
        return str(self)
        
    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        import shutil
        shutil.copy(str(self), output_path)

# ================= 节点 1: 加载视频 =================
class 加载视频_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "文件路径": ("STRING", {"default": "example.mp4", "multiline": False, "label": "文件名"}),
            },
            "optional": {
                "当前时间": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01, "label": "当前时间(秒)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "VIDEO")
    RETURN_NAMES = ("当前帧", "图像序列", "音频", "视频")
    OUTPUT_NODE = True
    FUNCTION = "load_video"
    CATEGORY = "💬 AI人工智能/加载视频"
    DESCRIPTION = "加载视频并支持在节点上预览当前帧，可输出单帧、完整序列及音频。"

    def load_video(self, 文件路径, 当前时间):
        path = 文件路径.strip().strip('"')
        if not os.path.isabs(path):
            path = folder_paths.get_annotated_filepath(path) or os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到视频: {path}")

        current_frame = extract_video_frames(path, 当前时间, 当前时间, single_frame=True)
        full_sequence = extract_video_frames(path, 0, 0, single_frame=False)
        audio = extract_audio(path, 0, 0)
        video_obj = QwenVideoOutput(path)

        return (current_frame, full_sequence, audio, video_obj)

    @classmethod
    def IS_CHANGED(s, 文件路径, 当前时间):
        return hashlib.md5(f"{文件路径}_{当前时间}".encode()).hexdigest()

# ================= 节点 2: 裁剪视频 =================
class 裁剪视频_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "文件路径": ("STRING", {"default": "example.mp4", "multiline": False, "label": "文件名"}),
            },
            "optional": {
                "开始时间": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01, "label": "开始时间(秒)"}),
                "持续时间": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01, "label": "持续时间(0=全长)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "VIDEO")
    RETURN_NAMES = ("图像序列", "音频", "视频")
    OUTPUT_NODE = True
    FUNCTION = "crop_video"
    CATEGORY = "💬 AI人工智能/加载视频"
    DESCRIPTION = "预览并裁剪视频片段，输出指定时间段的图像序列、音频及裁剪后的MP4文件。"

    def crop_video(self, 文件路径, 开始时间, 持续时间):
        path = 文件路径.strip().strip('"')
        if not os.path.isabs(path):
            path = folder_paths.get_annotated_filepath(path) or os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到视频: {path}")

        cropped_sequence = extract_video_frames(path, 开始时间, 开始时间 + 持续时间, single_frame=False)
        cropped_audio = extract_audio(path, 开始时间, 开始时间 + 持续时间)
        
        params_hash = hashlib.md5(f"{path}_{开始时间}_{持续时间}".encode("utf-8")).hexdigest()
        cropped_mp4_name = f"cropped_{params_hash}.mp4"
        cropped_mp4_path = os.path.join(folder_paths.get_temp_directory(), cropped_mp4_name)
        
        if not os.path.exists(cropped_mp4_path):
            try:
                cmd = ["ffmpeg", "-y", "-ss", str(开始时间), "-i", path]
                if 持续时间 > 0:
                    cmd.extend(["-t", str(持续时间)])
                cmd.extend(["-c", "copy", cropped_mp4_path])
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[裁剪视频] 尝试生成独立 MP4 失败: {e}")
                cropped_mp4_path = path

        video_obj = QwenVideoOutput(cropped_mp4_path)
        return (cropped_sequence, cropped_audio, video_obj)

    @classmethod
    def IS_CHANGED(s, 文件路径, 开始时间, 持续时间):
        return hashlib.md5(f"{文件路径}_{开始时间}_{持续时间}".encode()).hexdigest()