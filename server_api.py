import server
from aiohttp import web
import tkinter as tk
from tkinter import filedialog
import os
import folder_paths

# ================= API 扩展: 文件浏览与流媒体 =================

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

@server.PromptServer.instance.routes.post("/qwen/browse_file")
async def browse_file(request):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        # 允许选择媒体文件
        file_path = filedialog.askopenfilename(
            filetypes=[("Media Files", "*.mp4 *.mov *.avi *.mkv *.webm *.wav *.mp3 *.flac *.m4a *.ogg"), ("All Files", "*.*")]
        )
        root.destroy()
        if file_path:
            return web.json_response({"path": file_path.replace("\\", "/")})
        return web.json_response({"path": ""})
    except Exception as e:
        return web.json_response({"error": str(e)})

# ================= 专属流媒体接口 (用于视频播放) =================
@server.PromptServer.instance.routes.get("/qwen/view_media")
async def view_media(request):
    try:
        file_path = request.query.get("path", "")
        if not file_path:
            return web.Response(status=400, text="No path provided")
        
        # 兼容相对路径和绝对路径
        if not os.path.isabs(file_path):
            file_path = folder_paths.get_annotated_filepath(file_path) or os.path.abspath(file_path)
            
        if not os.path.isfile(file_path):
            return web.Response(status=404, text="File not found")
            
        # FileResponse 自动处理 Accept-Ranges，完美支持 HTML5 Video 拖拽进度条
        return web.FileResponse(file_path)
    except Exception as e:
        return web.Response(status=500, text=str(e))