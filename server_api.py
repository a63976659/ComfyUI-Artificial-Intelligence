import server
from aiohttp import web
import tkinter as tk
from tkinter import filedialog
import os
import folder_paths
import cv2

# ================= 1. 浏览文件夹 API =================
@server.PromptServer.instance.routes.post("/qwen/browse_folder")
async def browse_folder(request):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory()
        root.destroy()
        if folder_path:
            return web.json_response({"path": folder_path.replace("\\", "/")})
        return web.json_response({"path": ""})
    except Exception as e:
        return web.json_response({"error": str(e)})

# ================= 2. 浏览文件 API =================
@server.PromptServer.instance.routes.post("/qwen/browse_file")
async def browse_file(request):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            filetypes=[("Media Files", "*.mp4 *.mov *.avi *.mkv *.webm *.wav *.mp3 *.flac *.m4a *.ogg"), ("All Files", "*.*")]
        )
        root.destroy()
        if file_path:
            return web.json_response({"path": file_path.replace("\\", "/")})
        return web.json_response({"path": ""})
    except Exception as e:
        return web.json_response({"error": str(e)})

# ================= 3. 专属流媒体 API (支持 FLAC/MKV 等) =================
@server.PromptServer.instance.routes.get("/qwen/view_media")
async def view_media(request):
    try:
        file_path = request.query.get("path", "")
        if not file_path:
            return web.Response(status=400, text="No path provided")
        
        if not os.path.isabs(file_path):
            file_path = folder_paths.get_annotated_filepath(file_path) or os.path.abspath(file_path)
            
        if not os.path.isfile(file_path):
            return web.Response(status=404, text="File not found")
            
        # 强制 MIME 类型识别，解决浏览器拒绝播放 flac/mkv 等格式的问题
        ext = file_path.split('.')[-1].lower()
        mime_type = None
        
        if ext == "flac": mime_type = "audio/flac"
        elif ext == "ogg": mime_type = "audio/ogg"
        elif ext == "wav": mime_type = "audio/wav"
        elif ext == "mp3": mime_type = "audio/mpeg"
        elif ext == "m4a": mime_type = "audio/mp4"
        elif ext == "mp4": mime_type = "video/mp4"
        elif ext == "webm": mime_type = "video/webm"
        elif ext == "mkv": mime_type = "video/x-matroska"
        elif ext == "mov": mime_type = "video/quicktime"
        elif ext == "avi": mime_type = "video/x-msvideo"
            
        response = web.FileResponse(file_path)
        if mime_type:
            response.content_type = mime_type
            
        return response
    except Exception as e:
        return web.Response(status=500, text=str(e))

# ================= 4. 获取视频/音频元数据 API (FPS) =================
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