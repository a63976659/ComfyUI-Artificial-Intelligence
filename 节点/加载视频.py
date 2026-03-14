import os
import cv2
import torch
import numpy as np
import torchaudio
import folder_paths
import hashlib
import subprocess
import server
from aiohttp import web

# ================= 新增 API: 获取视频元数据 (FPS) =================
@server.PromptServer.instance.routes.get("/qwen/video_metadata")
async def qwen_video_metadata(request):
    file_path = request.query.get("path", "")
    if not file_path:
        return web.json_response({"fps": 0})
    
    if not os.path.isabs(file_path):
        file_path = folder_paths.get_annotated_filepath(file_path) or os.path.abspath(file_path)
        
    if not os.path.isfile(file_path):
        return web.json_response({"fps": 0})
        
    try:
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return web.json_response({"fps": fps, "total_frames": total_frames})
    except Exception as e:
        return web.json_response({"fps": 0, "error": str(e)})


def get_video_metadata(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total_frames / fps if fps > 0 else 0
    return fps, total_frames, duration

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
        waveform, sample_rate = torchaudio.load(path)
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate) if end_time > 0 else waveform.shape[1]
        
        if start_frame >= waveform.shape[1]:
            start_frame = 0
            
        cropped_waveform = waveform[:, start_frame:end_frame]
        return {"waveform": cropped_waveform.unsqueeze(0) if cropped_waveform.dim() == 2 else cropped_waveform, "sample_rate": sample_rate}
    except Exception as e:
        print(f"[视频音频提取] 失败或无音频: {e}")
        return {"waveform": torch.zeros((1, 1, 1024)), "sample_rate": 44100}

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

        return (current_frame, full_sequence, (audio,), path)

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
                cmd = [
                    "ffmpeg", "-y", "-i", path, 
                    "-ss", str(开始时间), 
                    "-t", str(持续时间) if 持续时间 > 0 else "9999",
                    "-c", "copy",
                    cropped_mp4_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[裁剪视频] 尝试生成独立 MP4 失败: {e}")
                cropped_mp4_path = path

        return (cropped_sequence, (cropped_audio,), cropped_mp4_path)

    @classmethod
    def IS_CHANGED(s, 文件路径, 开始时间, 持续时间):
        return hashlib.md5(f"{文件路径}_{开始时间}_{持续时间}".encode()).hexdigest()
