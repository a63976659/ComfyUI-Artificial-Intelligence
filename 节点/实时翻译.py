# -*- coding: utf-8 -*-
"""
实时翻译节点 (隔离环境子进程运行)
支持三家国产模型: 腾讯 Hunyuan-MT-7B / 字节 Seed-X-PPO-7B / 阿里 Qwen 系列，
翻译过程通过流式协议逐段推送到前端，在节点上打字机式实时显示。
只做语音翻译: 必须接入音频输入 (如麦克风录音节点)，内部先经 Qwen3-ASR 识别再翻译。
"""
import os
import re
import time
import uuid
import asyncio
import tempfile
import threading

import soundfile as sf
from aiohttp import web
from server import PromptServer

from .utils import resolve_llm_model, resolve_tts_model
from .隔离环境 import run_worker, get_manager
from .语音识别 import ASR_MODELS
from .累加文本 import push_append

# ================= 模型限定注册表 =================
# 固定下拉列表 (不扫描本地目录)，每个模型绑定各自的 prompt 风格，防止选错无法生效
TRANSLATE_MODELS = {
    "Hunyuan-MT-7B": {"style": "hunyuan", "repo": "tencent/Hunyuan-MT-7B"},
    "Seed-X-PPO-7B": {"style": "seedx", "repo": "ByteDance-Seed/Seed-X-PPO-7B"},
    "Qwen2.5-7B-Instruct": {"style": "qwen", "repo": "Qwen/Qwen2.5-7B-Instruct"},
    "Qwen2.5-3B-Instruct": {"style": "qwen", "repo": "Qwen/Qwen2.5-3B-Instruct"},
    "Qwen2.5-1.5B-Instruct": {"style": "qwen", "repo": "Qwen/Qwen2.5-1.5B-Instruct"},
    "Qwen3-4B-Instruct-2507": {"style": "qwen", "repo": "Qwen/Qwen3-4B-Instruct-2507"},
}

# ================= 目标语言表 =================
# 显示名 -> (英文名 [Hunyuan/Qwen 用], Seed-X 语言标签 [无则 None])
TARGET_LANGUAGES = {
    "中文": ("Chinese", "zh"),
    "英文": ("English", "en"),
    "日文": ("Japanese", "ja"),
    "韩文": ("Korean", "ko"),
    "法文": ("French", "fr"),
    "德文": ("German", "de"),
    "西班牙语": ("Spanish", "es"),
    "俄语": ("Russian", "ru"),
    "阿拉伯语": ("Arabic", "ar"),
    "葡萄牙语": ("Portuguese", "pt"),
    "意大利语": ("Italian", "it"),
    "泰语": ("Thai", "th"),
    "印地语": ("Hindi", None),
    "越南语": ("Vietnamese", "vi"),
    "印尼语": ("Indonesian", "id"),
    "荷兰语": ("Dutch", "nl"),
    "土耳其语": ("Turkish", "tr"),
    "马来语": ("Malay", "ms"),
    "芬兰语": ("Finnish", "fi"),
    "波兰语": ("Polish", "pl"),
    "乌克兰语": ("Ukrainian", "uk"),
}

STREAM_EVENT = "ai_realtime_translate"


def _save_audio_to_temp(audio_input):
    """将 ComfyUI 的音频 Tensor 存为临时 WAV 文件，供 ASR 子进程读取 (与语音识别节点一致)"""
    waveform = audio_input["waveform"]
    sample_rate = audio_input["sample_rate"]

    wav_tensor = waveform[0] if waveform.dim() == 3 else waveform
    if wav_tensor.shape[0] > wav_tensor.shape[1]:
        wav_tensor = wav_tensor.t()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.close()
    # soundfile 需要 (frames, channels)，而 ComfyUI 音频为 (channels, frames)
    sf.write(temp_file.name, wav_tensor.t().cpu().numpy(), int(sample_rate))
    return temp_file.name


def _contains_chinese(text):
    """判断文本是否包含中文字符 (用于选择 Hunyuan-MT 的中外/外外互译模板)"""
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def _build_payload(style, 原文, 目标语言, 最大生成长度):
    """按模型风格构造工作进程请求体"""
    lang_en, seedx_tag = TARGET_LANGUAGES[目标语言]

    if style == "hunyuan":
        # Hunyuan-MT 官方模板: 中外互译用中文指令，外外互译用英文指令
        if 目标语言 == "中文" or _contains_chinese(原文):
            content = f"把下面的文本翻译成{目标语言}，不要额外解释。\n\n{原文}"
        else:
            content = f"Translate the following segment into {lang_en}, without additional explanation.\n\n{原文}"
        return {
            "messages": [{"role": "user", "content": content}],
            "prompt_style": "hunyuan",
            "max_new_tokens": 最大生成长度,
        }

    if style == "seedx":
        # Seed-X 官方格式: 原始 prompt 结尾必须带目标语言标签 (如 <zh>)
        if seedx_tag is None:
            raise ValueError(
                f"Seed-X-PPO-7B 不支持目标语言\"{目标语言}\"，"
                "请更换目标语言或改用 Hunyuan-MT-7B / Qwen 模型"
            )
        prompt = f"Translate the following sentence into {lang_en}:\n{原文} <{seedx_tag}>"
        return {
            "prompt": prompt,
            "prompt_style": "seedx",
            "max_new_tokens": 最大生成长度,
        }

    # qwen: 沿用智能翻译节点的纯净翻译指令
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional translator. Translate the following text "
                    f"directly without explanation. Target Language: {lang_en}."
                ),
            },
            {"role": "user", "content": 原文},
        ],
        "prompt_style": "qwen",
        "max_new_tokens": 最大生成长度,
    }


class 实时翻译_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型名称": (list(TRANSLATE_MODELS.keys()), {"default": "Hunyuan-MT-7B"}),
                "目标语言": (list(TARGET_LANGUAGES.keys()), {"default": "英文"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "最大生成长度": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "音频": ("AUDIO",),
                "识别模型": (ASR_MODELS, {"default": "Qwen3-ASR-0.6B"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("翻译结果", "识别结果")
    FUNCTION = "translate"
    CATEGORY = "💬 AI人工智能/实时翻译"
    DESCRIPTION = (
        "实时翻译节点 (隔离进程运行)：支持腾讯 Hunyuan-MT / 字节 Seed-X / 阿里 Qwen 三家国产模型，"
        "翻译过程在节点上打字机式实时显示。本节点专做语音翻译：请把音频接到\"音频\"输入 "
        "(如 🎙️ 麦克风录音节点)，内部先经 Qwen3-ASR 识别再翻译。真·实时模式：连接麦克风录音节点后，"
        "点击录音节点上的\"🌐 开始实时翻译\"按钮，无需点击运行，边说边自动断句翻译，译文持续显示在本节点上。"
    )

    def translate(self, 模型名称, 目标语言, 自动下载模型, 最大生成长度,
                  运行后立即卸载=True, 音频=None, 识别模型="Qwen3-ASR-0.6B", unique_id=None):
        # 1. 语音识别得到原文 (本节点只做语音翻译，必须接入音频)
        if 音频 is None:
            raise ValueError("请先把音频连接到\"音频\"输入 (如 🎙️ 麦克风录音节点)，本节点只做语音翻译")
        asr_model_path = resolve_tts_model(识别模型, 自动下载模型, source="ModelScope")
        temp_path = _save_audio_to_temp(音频)
        try:
            asr_resp = run_worker(
                "asr",
                {
                    "model_path": asr_model_path,
                    "aligner_path": None,
                    "audio_path": temp_path,
                    "language": None,
                    "context": None,
                    "return_time_stamps": False,
                },
                unload_after=运行后立即卸载,
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        原文 = asr_resp.get("text", "").strip()
        if not 原文:
            raise ValueError("语音识别结果为空，请检查音频内容是否包含语音")
        print(f"[实时翻译] 语音识别原文: {原文}")

        # 2. 定位/下载翻译模型 (仅路径，加载在隔离子进程中完成)
        model_info = TRANSLATE_MODELS[模型名称]
        payload = _build_payload(model_info["style"], 原文, 目标语言, 最大生成长度)
        payload["model_path"] = resolve_llm_model(model_info["repo"], 自动下载模型)
        payload["stream"] = True

        # 3. 流式推理: 增量文本实时推送到前端节点显示区
        acc = []

        def on_stream(msg):
            delta = msg.get("delta", "")
            if not delta:
                return
            acc.append(delta)
            if unique_id is not None:
                PromptServer.instance.send_sync(
                    STREAM_EVENT, {"node": unique_id, "text": "".join(acc)}
                )

        resp = run_worker("llm", payload, unload_after=运行后立即卸载, on_stream=on_stream)
        return (resp.get("content", "").strip(), 原文)


# ================= 实时会话 (绕过图执行的持续翻译) =================
# ComfyUI 的节点连线只在"队列执行"时传一次数据，无法承载持续音频流。
# 参考 whisper_streaming / RealtimeSTT 等开源实时方案的通用架构:
#   前端持续采集麦克风 -> 能量 VAD 自动断句 -> 每句作为独立分段上传 ->
#   后端常驻子进程 ASR + 流式翻译 -> send_sync 推送到翻译节点显示区。
# 会话期间 asr/llm 子进程保持常驻 (unload_after=False)，停止时按需卸载。

_DISPLAY_RECENT = 3  # 显示区只保留最近几句 (早期句子仍完整追加到“持续填充文本”节点)

_SESSIONS = {}
_SESSIONS_LOCK = threading.Lock()
_SESSION_TTL_SECONDS = 3600  # 页面被直接关闭等异常场景下，过期会话自动清理

_routes = PromptServer.instance.routes


def _push_display(sess):
    """把会话累计文本推送到前端翻译节点显示区 (只展示最近 _DISPLAY_RECENT 句)"""
    recent = sess["lines"][-_DISPLAY_RECENT:] if sess["lines"] else []
    text = "\n\n".join(recent) if recent else "🎙️ 聆听中，请说话..."
    PromptServer.instance.send_sync(STREAM_EVENT, {"node": sess["node_id"], "text": text})


def _prune_sessions():
    now = time.time()
    with _SESSIONS_LOCK:
        for sid in [k for k, v in _SESSIONS.items() if now - v["last_active"] > _SESSION_TTL_SECONDS]:
            _SESSIONS.pop(sid, None)


def _process_utterance(sess, audio_path):
    """单个断句分段的完整处理链: 常驻 ASR 识别 -> 常驻 LLM 流式翻译 -> 增量推送"""
    with sess["lock"]:  # 会话内分段严格串行，保证句子顺序
        asr_resp = run_worker(
            "asr",
            {
                "model_path": sess["asr_path"],
                "aligner_path": None,
                "audio_path": audio_path,
                "language": None,
                "context": None,
                "return_time_stamps": False,
            },
            unload_after=False,
        )
        source = asr_resp.get("text", "").strip()
        if not source:
            return {"skipped": True}

        sess["lines"].append(f"🗣 {source}\n➜ ")
        _push_display(sess)

        payload = _build_payload(sess["style"], source, sess["target_lang"], sess["max_new_tokens"])
        payload["model_path"] = sess["model_path"]
        payload["stream"] = True

        acc = []

        def on_stream(msg):
            delta = msg.get("delta", "")
            if delta:
                acc.append(delta)
                sess["lines"][-1] = f"🗣 {source}\n➜ {''.join(acc)}"
                _push_display(sess)

        resp = run_worker("llm", payload, unload_after=False, on_stream=on_stream)
        translation = resp.get("content", "").strip()
        sess["lines"][-1] = f"🗣 {source}\n➜ {translation}"
        _push_display(sess)
        # 连有"持续填充文本"节点时，每句把原文 + 译文成对追加，实现双语同步记录
        push_append(sess.get("sink_id"), source, translation)
        return {"source": source, "translation": translation}


@_routes.post("/qwen/realtime/start")
async def _realtime_start(request):
    """创建实时翻译会话: 校验语言支持并解析模型路径 (可触发自动下载)"""
    try:
        data = await request.json()
        model_info = TRANSLATE_MODELS[data["model"]]
        目标语言 = data.get("target_lang", "英文")
        if 目标语言 not in TARGET_LANGUAGES:
            raise ValueError(f"不支持的目标语言: {目标语言}")
        if model_info["style"] == "seedx" and TARGET_LANGUAGES[目标语言][1] is None:
            raise ValueError(
                f"Seed-X-PPO-7B 不支持目标语言\"{目标语言}\"，"
                "请更换目标语言或改用 Hunyuan-MT-7B / Qwen 模型"
            )

        def prepare():
            model_path = resolve_llm_model(model_info["repo"], bool(data.get("auto_download")))
            asr_path = resolve_tts_model(data.get("asr_model", "Qwen3-ASR-0.6B"),
                                         bool(data.get("auto_download")), source="ModelScope")
            # 预热: 把 ASR + LLM 权重真正加载进常驻子进程再返回，
            # 使前端"正在加载模型"标签持续到模型就绪，避免首句才加载造成"已在翻译中"的错觉
            run_worker("asr", {"action": "warmup", "model_path": asr_path, "aligner_path": None},
                       unload_after=False)
            run_worker("llm", {"action": "warmup", "model_path": model_path}, unload_after=False)
            return model_path, asr_path

        model_path, asr_path = await asyncio.get_running_loop().run_in_executor(None, prepare)
    except Exception as e:
        print(f"[实时翻译] 会话创建失败: {e}")
        return web.json_response({"error": str(e)}, status=500)

    _prune_sessions()
    session_id = uuid.uuid4().hex
    sess = {
        "node_id": str(data.get("node_id", "")),
        "sink_id": str(data.get("sink_id", "") or ""),  # 可选的持续填充文本节点 id
        "style": model_info["style"],
        "model_path": model_path,
        "asr_path": asr_path,
        "target_lang": 目标语言,
        "max_new_tokens": int(data.get("max_new_tokens", 1024)),
        "lines": [],
        "lock": threading.Lock(),
        "last_active": time.time(),
    }
    with _SESSIONS_LOCK:
        _SESSIONS[session_id] = sess
    _push_display(sess)
    print(f"[实时翻译] 实时会话已创建: {data['model']} -> {目标语言}")
    return web.json_response({"session": session_id})


@_routes.post("/qwen/realtime/chunk")
async def _realtime_chunk(request):
    """处理一个 VAD 断句分段: ASR 识别 -> 流式翻译 -> 推送显示"""
    session_id = None
    temp_path = None
    try:
        reader = await request.multipart()
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "session":
                session_id = (await field.text()).strip()
            elif field.name == "audio":
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                while True:
                    chunk = await field.read_chunk()
                    if not chunk:
                        break
                    temp.write(chunk)
                temp.close()
                temp_path = temp.name

        with _SESSIONS_LOCK:
            sess = _SESSIONS.get(session_id)
        if sess is None:
            return web.json_response({"error": "会话不存在或已结束"}, status=400)
        if temp_path is None:
            return web.json_response({"error": "缺少音频数据"}, status=400)
        sess["last_active"] = time.time()

        # 推理阻塞放线程池，避免卡住 ComfyUI 的 aiohttp 事件循环
        result = await asyncio.get_running_loop().run_in_executor(
            None, _process_utterance, sess, temp_path)
        return web.json_response(result)
    except Exception as e:
        print(f"[实时翻译] 分段处理失败: {e}")
        return web.json_response({"error": str(e)}, status=500)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@_routes.post("/qwen/realtime/stop")
async def _realtime_stop(request):
    """结束会话; unload=True 时立即卸载 asr/llm 子进程释放显存"""
    try:
        data = await request.json()
        with _SESSIONS_LOCK:
            sess = _SESSIONS.pop(data.get("session", ""), None)
        # 显式推送停止状态，避免显示区停留在"聆听中"造成未停止的错觉
        if sess is not None:
            recent = sess["lines"][-_DISPLAY_RECENT:] if sess["lines"] else []
            suffix = "\n\n⏹ 实时翻译已停止" if recent else "⏹ 实时翻译已停止"
            PromptServer.instance.send_sync(STREAM_EVENT, {
                "node": sess["node_id"],
                "text": ("\n\n".join(recent) if recent else "") + suffix,
            })
        if data.get("unload"):
            def unload():
                for family in ("asr", "llm"):
                    manager = get_manager(family)
                    with manager.lock:
                        manager._shutdown()
            await asyncio.get_running_loop().run_in_executor(None, unload)
        print("[实时翻译] 实时会话已结束")
        return web.json_response({"ok": True})
    except Exception as e:
        print(f"[实时翻译] 会话结束异常: {e}")
        return web.json_response({"error": str(e)}, status=500)
