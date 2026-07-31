# -*- coding: utf-8 -*-
"""
通用模型隔离环境底座
每个模型家族使用独立虚拟环境 + 独立推理子进程，互不干扰:
- 虚拟环境通过 venv --system-site-packages 创建，共享主环境 torch 等大件，
  仅在环境内独立安装各自锁定版本的模型运行时依赖 (如 transformers)
- 推理在子进程 (工作进程/*.py) 中完成，主进程不导入任何模型运行时库
- 子进程退出即彻底释放显存；常驻模式下空闲超时自动卸载
- 新增模型支持只需: 在 ENV_SPECS 注册环境规格 + 编写对应的工作进程脚本
"""
import os
import sys
import json
import atexit
import threading
import subprocess

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS_DIR = os.path.join(PLUGIN_ROOT, "工作进程")

# 与 工作进程/协议.py 中的定义保持一致
PROTOCOL_PREFIX = "##AI_WORKER_JSON##"
STREAM_PREFIX = "##AI_WORKER_STREAM##"
IDLE_TIMEOUT_SECONDS = 600  # 常驻模式下空闲 10 分钟自动卸载

# 环境规格注册表: 新增模型家族时在此登记
# env_dir 使用英文命名 (venv/pip 工具链对非 ASCII 路径兼容性差，属必要例外)
ENV_SPECS = {
    "llm": {
        "label": "Qwen文本",
        "env_dir": "llm_env",
        "packages": ["transformers==4.57.3"],
        "worker": "Qwen文本推理.py",
    },
    "tts": {
        "label": "Qwen语音",
        "env_dir": "tts_env",
        "packages": ["transformers==4.57.3", "qwen-tts>=0.1.1"],
        "worker": "Qwen语音推理.py",
    },
    "asr": {
        "label": "Qwen语音识别",
        "env_dir": "asr_env",
        "packages": ["transformers==4.57.6", "qwen-asr==0.0.6"],
        "worker": "Qwen语音识别推理.py",
    },
    "gemma": {
        "label": "Gemma",
        "env_dir": "gemma_env",
        "packages": ["transformers>=5.14,<6"],
        "worker": "Gemma多模态推理.py",
    },
}

SPEC_MARKER_NAME = "环境规格.json"

# ================= 隔离环境自举 =================

def _use_pkgs_fallback():
    """官方便携版内嵌的 embedded Python 缺少 venv/ensurepip 模块，无法创建虚拟环境。
    此时降级为 pip --target 安装到独立目录，并由工作进程注入 sys.path，实现同等隔离效果。"""
    import importlib.util
    return (importlib.util.find_spec("venv") is None
            or importlib.util.find_spec("ensurepip") is None)


def _pkgs_dir(env_dir):
    return os.path.join(env_dir, "pkgs")


def _env_paths(family):
    spec = ENV_SPECS[family]
    env_dir = os.path.join(PLUGIN_ROOT, spec["env_dir"])
    if _use_pkgs_fallback():
        # 降级模式: 直接用主环境 Python 运行工作进程，依赖目录经 AI_WORKER_PKGS 注入
        return env_dir, sys.executable
    # 虚拟环境内 Python 路径: Windows 为 Scripts\python.exe，macOS/Linux 为 bin/python
    if os.name == "nt":
        python_exe = os.path.join(env_dir, "Scripts", "python.exe")
    else:
        python_exe = os.path.join(env_dir, "bin", "python")
    return env_dir, python_exe


def ensure_env(family):
    """确保隔离环境存在且依赖与规格一致，首次使用/规格变更时自动创建或更新"""
    spec = ENV_SPECS[family]
    env_dir, python_exe = _env_paths(family)
    marker_path = os.path.join(env_dir, SPEC_MARKER_NAME)

    if _use_pkgs_fallback():
        os.makedirs(env_dir, exist_ok=True)
    elif not os.path.exists(python_exe):
        print(f"[{spec['label']}] 首次使用，正在创建隔离环境 {spec['env_dir']} "
              f"(共享主环境 torch，仅独立安装 {', '.join(spec['packages'])})...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "--system-site-packages", env_dir])
        except Exception as e:
            raise RuntimeError(
                f"隔离环境创建失败: {e}\n可手动执行:\n"
                f"  \"{sys.executable}\" -m venv --system-site-packages \"{env_dir}\""
            )

    # 依赖规格比对: 首次创建或 ENV_SPECS 中的版本要求变化时 (重新) 安装
    installed_packages = None
    if os.path.exists(marker_path):
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                installed_packages = json.load(f).get("packages")
        except Exception:
            pass

    if installed_packages != spec["packages"]:
        print(f"[{spec['label']}] 正在安装隔离环境依赖: {', '.join(spec['packages'])}...")
        if _use_pkgs_fallback():
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade",
                   "--target", _pkgs_dir(env_dir)] + spec["packages"]
        else:
            cmd = [python_exe, "-m", "pip", "install"] + spec["packages"]
        try:
            subprocess.check_call(cmd)
        except Exception as e:
            pkgs = " ".join(f'"{p}"' for p in spec["packages"])
            raise RuntimeError(
                f"隔离环境依赖安装失败: {e}\n可手动执行:\n"
                f"  \"{cmd[0]}\" -m pip install {' '.join(cmd[3:-len(spec['packages'])])} {pkgs}"
            )
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump({"packages": spec["packages"]}, f, ensure_ascii=False, indent=2)
        print(f"[{spec['label']}] 隔离环境就绪")

# ================= 通用子进程管理器 =================

class IsolatedWorkerManager:
    """管理某个模型家族的推理子进程: 启动、请求、按需卸载、空闲超时"""

    def __init__(self, family):
        self.family = family
        self.label = ENV_SPECS[family]["label"]
        self.proc = None
        self.lock = threading.Lock()
        self.idle_timer = None
        atexit.register(self._shutdown)

    def _spawn(self):
        env_dir, python_exe = _env_paths(self.family)
        worker_script = os.path.join(WORKERS_DIR, ENV_SPECS[self.family]["worker"])
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["AI_WORKER_TAG"] = self.label
        if _use_pkgs_fallback():
            # 降级模式: 依赖目录路径传给工作进程，由 协议.py 注入 sys.path
            # (embedded Python 会忽略 PYTHONPATH，不能用环境变量方式)
            env["AI_WORKER_PKGS"] = _pkgs_dir(env_dir)
        print(f"[{self.label}] 启动隔离推理子进程...")
        # stderr 不重定向: 子进程日志/下载进度直接显示在 ComfyUI 控制台
        self.proc = subprocess.Popen(
            [python_exe, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            cwd=PLUGIN_ROOT,
            env=env,
        )

    def _cancel_idle_timer(self):
        if self.idle_timer is not None:
            self.idle_timer.cancel()
            self.idle_timer = None

    def _shutdown(self):
        """结束子进程并释放显存"""
        self._cancel_idle_timer()
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"action": "exit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        print(f"[{self.label}] 推理子进程已卸载，显存已释放")
        self.proc = None

    def _shutdown_idle(self):
        with self.lock:
            print(f"[{self.label}] 空闲超过 {IDLE_TIMEOUT_SECONDS // 60} 分钟")
            self._shutdown()

    def request(self, payload, unload_after, on_stream=None):
        with self.lock:
            self._cancel_idle_timer()
            if self.proc is None or self.proc.poll() is not None:
                self._spawn()
            try:
                self.proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
                self.proc.stdin.flush()

                resp = None
                while True:
                    line = self.proc.stdout.readline()
                    if line == "":
                        raise RuntimeError(
                            f"{self.label} 子进程异常退出 (常见原因: 显存不足 / 模型文件损坏)，"
                            "详细错误请查看控制台日志"
                        )
                    line = line.strip()
                    if line.startswith(PROTOCOL_PREFIX):
                        resp = json.loads(line[len(PROTOCOL_PREFIX):])
                        break
                    elif line.startswith(STREAM_PREFIX):
                        if on_stream is not None:
                            try:
                                on_stream(json.loads(line[len(STREAM_PREFIX):]))
                            except Exception as e:
                                print(f"[{self.label}] 流式回调异常 (已忽略): {e}")
                    elif line:
                        print(f"[{self.label}] {line}")
            except Exception:
                self._shutdown()
                raise

            if unload_after:
                self._shutdown()
            else:
                self.idle_timer = threading.Timer(IDLE_TIMEOUT_SECONDS, self._shutdown_idle)
                self.idle_timer.daemon = True
                self.idle_timer.start()

            if not resp.get("ok"):
                raise RuntimeError(f"{self.label} 推理失败: {resp.get('error', '未知错误')}")
            return resp


_MANAGERS = {}
_MANAGERS_LOCK = threading.Lock()


def get_manager(family):
    """按模型家族获取子进程管理器单例"""
    with _MANAGERS_LOCK:
        if family not in _MANAGERS:
            _MANAGERS[family] = IsolatedWorkerManager(family)
        return _MANAGERS[family]


def run_worker(family, payload, unload_after, on_stream=None):
    """公共入口: 确保隔离环境就绪 -> 发送请求 -> 返回响应

    on_stream: 可选回调，收到子进程的流式增量消息 (dict) 时被调用。
    """
    ensure_env(family)
    payload.setdefault("action", "generate")
    return get_manager(family).request(payload, unload_after=unload_after, on_stream=on_stream)
