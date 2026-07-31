# -*- coding: utf-8 -*-
"""
工作进程公共协议
运行环境: 各模型家族的隔离虚拟环境，不要在 ComfyUI 主进程中导入本文件。
协议: stdin 每行一个 JSON 请求; stdout 以 ##AI_WORKER_JSON## 前缀返回一行 JSON 响应;
      所有日志/进度输出到 stderr，避免污染协议通道。
"""
import os
import sys
import json
import traceback

# 便携版 (embedded Python) 降级模式: 主进程通过 AI_WORKER_PKGS 传入 pip --target 安装的
# 隔离依赖目录。embedded Python 会忽略 PYTHONPATH，必须在此显式注入，
# 且须先于 transformers 等库的首次导入 (各工作进程均最先导入本模块)。
_pkgs_dir = os.environ.get("AI_WORKER_PKGS")
if _pkgs_dir and os.path.isdir(_pkgs_dir) and _pkgs_dir not in sys.path:
    sys.path.insert(0, _pkgs_dir)

PROTOCOL_PREFIX = "##AI_WORKER_JSON##"
STREAM_PREFIX = "##AI_WORKER_STREAM##"
WORKER_TAG = os.environ.get("AI_WORKER_TAG", "AI工作进程")


def log(msg):
    print(f"[{WORKER_TAG}] {msg}", file=sys.stderr, flush=True)


def send(obj):
    print(PROTOCOL_PREFIX + json.dumps(obj, ensure_ascii=True), flush=True)


def send_stream(obj):
    """发送流式增量消息 (如逐段生成的文本)，不结束本次请求"""
    print(STREAM_PREFIX + json.dumps(obj, ensure_ascii=True), flush=True)


def main_loop(handlers):
    """通用请求循环。handlers: {action名: 处理函数(req) -> dict}，exit 指令内置。"""
    log(f"工作进程启动 (python {sys.version.split()[0]})")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            log(f"忽略无法解析的请求: {line[:100]}")
            continue

        action = req.get("action")
        if action == "exit":
            log("收到退出指令，进程结束")
            send({"ok": True, "bye": True})
            break
        elif action in handlers:
            try:
                send(handlers[action](req))
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        else:
            send({"ok": False, "error": f"未知指令: {action}"})
