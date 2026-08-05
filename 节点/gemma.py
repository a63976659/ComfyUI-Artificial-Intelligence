# -*- coding: utf-8 -*-
"""
Gemma 4 多模态节点 (隔离进程方案)
由于 qwen-tts 锁定 transformers==4.57.3，而 Gemma 4 需要 transformers>=5.14，
本节点通过独立虚拟环境 gemma_env 中的子进程 (工作进程/Gemma多模态推理.py) 完成推理:
- 主进程不导入新版 transformers，对其它节点零影响
- 子进程退出即彻底释放显存 (运行后立即卸载)
- 常驻模式下空闲超时自动退出
环境与子进程的通用管理见 节点/隔离环境.py
"""
import os
import uuid

import numpy as np
from PIL import Image

import folder_paths
from .utils import LLM_MODELS_DIR, HAS_MODELSCOPE
from .chat import SYSTEM_PROMPTS
from .隔离环境 import run_worker
from .H3提示词模版 import H3_PROMPT_TEMPLATES, H3_TEMPLATE_OPTIMIZE_PROMPT
from huggingface_hub import snapshot_download as hf_snapshot_download

if HAS_MODELSCOPE:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

DEFAULT_GEMMA_MODEL = "gemma-4-12B-it"

# 最大生成长度下拉选项: 上限按 Gemma 4 官方最大输出长度 32K 设定
GEMMA_NEW_TOKEN_OPTIONS = ["1024", "2048", "4096", "8192", "16384", "32768"]

# 抽帧档位: 快速/中等按固定帧数均匀抽帧, 详细抽取视频全部帧 (不降采样)
FRAME_SAMPLE_MODES = ["快速", "中等", "详细"]
FRAME_SAMPLE_COUNTS = {"快速": 8, "中等": 30}


def _frame_sample_indices(total, 视频分析):
    """按档位取帧下标: 详细档抽取全部帧, 其余按固定帧数均匀抽取"""
    count = total if 视频分析 == "详细" else min(FRAME_SAMPLE_COUNTS[视频分析], total)
    return np.linspace(0, total - 1, count).astype(int)

# ================= 模型下载 =================

def resolve_gemma_model(model_name, auto_download, source):
    """定位本地模型目录，不存在时按下载源自动下载"""
    target_folder = model_name.split("/")[-1] if "/" in model_name else model_name
    repo_id = model_name if "/" in model_name else f"google/{model_name}"

    for p in (os.path.join(LLM_MODELS_DIR, model_name), os.path.join(LLM_MODELS_DIR, target_folder)):
        if os.path.exists(p) and any(f.endswith(".safetensors") for f in os.listdir(p)):
            return p

    if not auto_download:
        raise FileNotFoundError(
            f"未找到模型 {model_name} (应位于 models/LLM/{target_folder})，"
            f"请开启\"自动下载模型\"或手动下载。"
        )

    download_path = os.path.join(LLM_MODELS_DIR, target_folder)
    if source == "ModelScope":
        if not HAS_MODELSCOPE:
            raise ImportError("请先安装 modelscope: pip install modelscope")
        print(f"[Gemma] 从 ModelScope 下载: {repo_id} -> {download_path}")
        ms_snapshot_download(model_id=repo_id, local_dir=download_path)
    elif source == "HF Mirror":
        print(f"[Gemma] 从 HF Mirror 下载: {repo_id} -> {download_path}")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        hf_snapshot_download(repo_id=repo_id, local_dir=download_path, resume_download=True, max_workers=4)
    else:  # HuggingFace 官方
        print(f"[Gemma] 从 HuggingFace 下载: {repo_id} -> {download_path}")
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        hf_snapshot_download(repo_id=repo_id, local_dir=download_path, resume_download=True, max_workers=4)
    return download_path


def get_installed_gemma_models():
    """扫描 models/LLM 下的 gemma-4 系列模型"""
    if not os.path.exists(LLM_MODELS_DIR):
        return []
    return sorted(
        d for d in os.listdir(LLM_MODELS_DIR)
        if os.path.isdir(os.path.join(LLM_MODELS_DIR, d)) and "gemma-4" in d.lower()
    )

# ================= 节点定义 =================

# 任务预设 (显示文本 -> 提示词, None 表示使用自定义问题)
TASK_PROMPTS = {
    "反推提示词 | 生成SD/Flux英文提示词": (
        "Describe this image as a detailed image-generation prompt in English for "
        "Stable Diffusion / Flux. Output only comma-separated tags and short phrases "
        "covering subject, appearance, style, lighting, composition and quality. "
        "Do not write full sentences or explanations."
    ),
    "详细描述 | 中文详细描述图片内容": "请用中文详细描述这张图片的内容，包括主体、背景、色彩、光线、构图和整体氛围。",
    "简短标题 | 一句话概括图片": "请用一句简短的中文概括这张图片的内容，不超过20个字。只输出这句话。",
    "OCR提取 | 提取图中所有文字": (
        "Extract all visible text from this image exactly as written. "
        "Preserve the original language and reading order. Output only the extracted text."
    ),
    "MiniMax H3 | 图生视频": (
        "你是MiniMax H3视频模型的提示词专家。请仔细观察这张参考图像，为MiniMax H3的图生视频功能"
        "（该图像将作为first_frame首帧传入）撰写一份高质量提示词。\n"
        "MiniMax H3提示词规则：\n"
        "1. 采用四段式结构：【参考素材说明】【核心创意】【画面过程描述】【整体要求补充】。\n"
        "2. 【参考素材说明】写明：@图片1（first_frame）为起始画面，描述其主体、场景、光影与色调。\n"
        "3. 【核心创意】写明“约N秒图生视频”（N取4~15），用一句话锁定主体、地点、事件、题材风格与运镜需求。\n"
        "4. 【画面过程描述】首帧图生视频H3不会自动切镜，全文保持一段连贯情节描述，不要使用【镜头N】/Shot结构；"
        "按时间点描述从首帧自然延续的动作过程（如0~2秒、2~5秒），每段写“正向：要呈现的动作与画面 - 反向：不要出现的问题”；"
        "台词长短须与对应时间段匹配，跨段台词要写明“接着上一段继续说”；画面中出现的文字、Logo、标题须给出原文；"
        "若用户要求中包含具体台词/对白/歌词，其内容必须一字不差地完整保留在输出提示词中并安排到对应时间段，"
        "不得省略、缩写、改写或概括。\n"
        "5. 【整体要求补充】包含：▍影像风格（胶片质感/色调/布光等，与图像风格一致）；"
        "▍声音设计（H3输出自带原生立体声：背景音乐、环境音效、人物台词与音色，具体到乐器与落点；不要背景音乐时明确写“非叙事性音乐：N/A”）；"
        "▍附加要求（转场方式、禁止元素如“不要字幕、不要水印”、一致性如“保持主体、场景、色调与输入图片一致”）。\n"
        "6. 描述尽量具体、直接、可视化，少用需要意会的比喻。\n"
        "7. 全文不超过7000字符，直接输出完整提示词，不要输出任何解释。\n"
        "若用户在自定义问题中附加了具体要求，请优先满足其要求。"
    ),
    "MiniMax H3 | 首尾帧视频": (
        "你是MiniMax H3视频模型的提示词专家。请仔细观察两张参考图像：【图像1】将作为first_frame首帧，"
        "【图像2】将作为last_frame尾帧。为MiniMax H3的首尾帧图生视频功能撰写一份高质量提示词。\n"
        "MiniMax H3提示词规则：\n"
        "1. 采用四段式结构：【参考素材说明】【核心创意】【画面过程描述】【整体要求补充】。\n"
        "2. 【参考素材说明】分别描述首帧画面（@图片1）与尾帧画面（@图片2）的主体、场景、光影、色调及两者的差异。\n"
        "3. 【核心创意】写明“约N秒视频”（N取4~15），用一句话锁定主体、地点、事件、题材风格，概括从首帧到尾帧的核心演变。\n"
        "4. 【画面过程描述】首尾帧模式H3不会自动切镜，只补两帧之间的动作、光影与声音，全文保持一段连贯演变描述，"
        "不要使用【镜头N】/Shot结构；按时间点写“正向：… - 反向：不要…”，演变须自然连贯、符合物理规律；"
        "台词长短须与对应时间段匹配；画面中出现的文字、Logo、标题须给出原文；"
        "若用户要求中包含具体台词/对白/歌词，其内容必须一字不差地完整保留在输出提示词中并安排到对应时间段，"
        "不得省略、缩写、改写或概括。\n"
        "5. 【整体要求补充】包含：▍一致性要求（保持首帧与尾帧的主体、场景、色调与输入图片一致）；"
        "▍声音设计（H3输出自带原生立体声：背景音乐、环境音效、台词音色，具体到乐器与落点；不要背景音乐时明确写“非叙事性音乐：N/A”）；"
        "▍附加要求（转场方式、禁止元素如“不要字幕、不要水印”）。\n"
        "6. 描述尽量具体、直接、可视化，少用需要意会的比喻。\n"
        "7. 全文不超过7000字符，直接输出完整提示词，不要输出任何解释。\n"
        "若用户在自定义问题中附加了具体要求，请优先满足其要求。"
    ),
    "自定义问题 | 使用下方自定义问题提问": None,
}

VISUAL_TOKEN_BUDGETS = ["70", "140", "280", "560", "1120"]

# MiniMax H3 提示词撰写系统指令 (仅供 Gemma 多模态对话节点使用，不影响千问文本对话节点)
MINIMAX_H3_SYSTEM_PROMPTS = {
    "MiniMax H3 | 文生视频": (
        "你是MiniMax H3视频模型的提示词专家。请根据用户的需求撰写一份MiniMax H3文生视频高质量提示词。\n"
        "MiniMax H3规则（无参考素材，纯文字生成视频，输出最高2K、24FPS、自带原生立体声，时长4~15秒）：\n"
        "1. 采用四段式结构：【参考素材说明】【核心创意】【画面过程描述】【整体要求补充】。\n"
        "2. 【参考素材说明】写：无参考素材（纯文字生成视频）。\n"
        "3. 【核心创意】写明“N秒视频”（N取4~15，可根据用户需求判断），用一句话锁定主体、地点、事件、题材风格与看点；"
        "需要特殊运镜时写明（一镜到底/慢动作/航拍等；环绕运镜建议写“truck left+pan right”而非直接说环绕），不写则默认切镜。\n"
        "4. 【画面过程描述】按时间点拆分镜（如0~3秒、3~5秒），写明景别（大全景交代空间+中景承载动作+特写强调细节的分层写法）、运镜、动作与音效；"
        "台词长短须与镜头长短对齐（口型问题的主要来源），跨镜台词写明“接着上个shot继续说”并标注说话人；"
        "画面中出现的文字、Logo、标题、按钮文案必须写出原文；切镜时写明切到什么景别、主体是哪个角色以保持跨镜头一致性；"
        "若用户需求中包含具体台词/对白/歌词，其内容必须一字不差地完整保留在输出提示词中并安排到对应镜头，"
        "不得省略、缩写、改写或概括。\n"
        "5. 【整体要求补充】包含：▍影像风格（胶片质感、色调、布光等专业影视词汇）；"
        "▍声音设计（H3输出自带原生立体声，写明背景音乐、环境音效、人物台词与音色、音画同步落点；不要背景音乐时明确写“非叙事性音乐：N/A”，不要与“要BGM”矛盾）；"
        "▍附加要求（转场方式如cut/fade/卡点切/快切、禁止元素如“不要字幕、不要水印”、一致性要求）。\n"
        "6. 描述尽量具体、直接、可视化，少写需要意会的比喻句。\n"
        "7. 全文不超过7000字符。只输出最终提示词本身，不要输出解释或寒暄。"
    ),
    "MiniMax H3 | 参考生视频": (
        "你是MiniMax H3视频模型的提示词专家。请全面分析用户提供的参考素材（消息中按【图像N】【视频N】【音频N】编号标注），"
        "结合用户需求，撰写一份MiniMax H3全能参考模式的高质量提示词。\n"
        "MiniMax H3规则（输出最高2K、24FPS、自带原生立体声，时长4~15秒）：\n"
        "1. 参考素材在提示词中用@图片N/@视频N/@音频N引用，编号与素材编号一一对应；"
        "给每个素材分配明确的参考任务（官方任务类型：人物参考锁定脸与形象/物体参考/场景参考/关键帧/音色参考/故事版/风格参考/构图参考/"
        "音频复用/音频部分复用/动作参考/运镜参考/视频编辑），"
        "如“@图片1：人物参考（锁定脸、发型、服装轮廓、气质）”“@视频1：运镜参考（锁定镜头节奏与剪辑风格）”；素材中想强保持的特征建议明确写出。\n"
        "2. 素材上限：图片≤9张、视频≤3段、音频≤3段且音频必须搭配图片或视频，混合总数≤12。\n"
        "3. 采用四段式结构：【参考素材说明】【核心创意】【画面过程描述】【整体要求补充】。\n"
        "4. 【参考素材说明】逐一列出各素材编号、参考维度与内容描述，并在结尾附加官方约束句："
        "“只参考指定维度，不直接复制参考图，不出现真实品牌、原参考图logo/标题/可识别文字。”\n"
        "5. 【核心创意】写明“N秒视频”（N取4~15），用一句话锁定主体、地点、事件、题材风格，概括参考素材如何共同构成画面；与素材直接关联的元素用@引用强调。\n"
        "6. 【画面过程描述】按时间点拆分镜，每段写“正向：… - 反向：不要…”，写明景别与运镜；"
        "台词长短须与镜头长短对齐，跨镜台词写明“接着上个shot继续说”；画面中出现的文字、Logo须给出原文；"
        "若用户需求中包含具体台词/对白/歌词，其内容必须一字不差地完整保留在输出提示词中并安排到对应镜头，"
        "不得省略、缩写、改写或概括。\n"
        "7. 【整体要求补充】包含：▍影像风格；▍声音设计（H3输出自带原生立体声：背景音乐、环境音效、人物台词与音色；"
        "音频参考素材用于音色克隆/复用时，若需保持歌词或对白内容，强烈建议把具体歌词文本写出；不要背景音乐时明确写“非叙事性音乐：N/A”）；"
        "▍附加要求（转场、禁止元素、一致性）。\n"
        "8. 描述尽量具体、直接、可视化，少写需要意会的比喻句。\n"
        "9. 全文不超过7000字符。只输出最终提示词本身，不要输出解释或寒暄。"
    ),
}

# Gemma 对话节点的系统指令列表 = 通用预设 + H3 预设 ("自定义"占位项保持末尾)
GEMMA_SYSTEM_PROMPTS = {k: v for k, v in SYSTEM_PROMPTS.items() if not k.startswith("自定义")}
GEMMA_SYSTEM_PROMPTS.update(MINIMAX_H3_SYSTEM_PROMPTS)
GEMMA_SYSTEM_PROMPTS.update({k: v for k, v in SYSTEM_PROMPTS.items() if k.startswith("自定义")})

# 多模态对话节点动态输入上限 (前端默认每类只显示 1 个槽位，连接后自动追加，见 web/Gemma多模态动态输入.js)
CHAT_MAX_IMAGE = 9
CHAT_MAX_VIDEO = 3
CHAT_MAX_AUDIO = 3


# ================= 临时文件辅助 =================

def _temp_path(suffix):
    tmp_dir = folder_paths.get_temp_directory()
    os.makedirs(tmp_dir, exist_ok=True)
    return os.path.join(tmp_dir, f"gemma_{uuid.uuid4().hex}{suffix}")


def _save_temp_image(img_tensor):
    """单帧 [H,W,C] float 张量存为临时 PNG，返回路径"""
    img_np = (np.clip(img_tensor.cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
    path = _temp_path(".png")
    Image.fromarray(img_np).save(path)
    return path


def _save_temp_audio(audio):
    """ComfyUI AUDIO ({'waveform','sample_rate'}) 存为临时 WAV，返回路径。
    使用 soundfile 写出，避免 torchaudio 的 torchcodec 后端依赖。"""
    import soundfile as sf
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    waveform = audio["waveform"]
    if waveform.dim() == 3:
        waveform = waveform[0]
    path = _temp_path(".wav")
    # soundfile 需要 (frames, channels)，waveform 为 (channels, frames)
    sf.write(path, waveform.t().cpu().numpy(), int(audio["sample_rate"]))
    return path


def _cleanup_files(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _run_gemma(模型名称, 自动下载模型, 下载源, payload, unload_after):
    """公共流程: 定位/下载模型 -> 隔离环境子进程推理"""
    payload["model_path"] = resolve_gemma_model(模型名称, 自动下载模型, 下载源)
    resp = run_worker("gemma", payload, unload_after=unload_after)
    return (resp.get("content", ""), resp.get("thinking", ""))


class Gemma_Image_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_gemma_models()
        all_models = sorted(set(installed + [DEFAULT_GEMMA_MODEL]))
        task_keys = list(TASK_PROMPTS.keys())
        return {
            "required": {
                "图像1": ("IMAGE",),
                "任务类型": (task_keys, {"default": task_keys[0]}),
                "自定义问题": ("STRING", {"multiline": True, "default": "", "placeholder": "任务类型选\"自定义问题\"时在此输入..."}),
                "模型名称": (all_models, {"default": DEFAULT_GEMMA_MODEL}),
                "量化方式": (["4bit", "8bit", "bf16"], {"default": "4bit"}),
                "视觉Token预算": (VISUAL_TOKEN_BUDGETS, {"default": "560"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "思考模式": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HF Mirror", "HuggingFace"], {"default": "ModelScope"}),
                "最大生成长度": (GEMMA_NEW_TOKEN_OPTIONS, {"default": "1024"}),
            },
            "optional": {
                "图像2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("回复内容", "思考过程")
    OUTPUT_NODE = True
    FUNCTION = "analyze"
    CATEGORY = "💬 AI人工智能/谷歌系列"
    DESCRIPTION = (
        "基于 Gemma 4 的图像理解节点 (隔离进程运行，不影响其它节点)。\n"
        "支持反推提示词、图像描述、OCR 和自定义提问。\n"
        "可选接入第二张图(图像2)，两张同接时自定义问题可分别引用【图像1】【图像2】(如对比两图)。\n"
        "OCR 建议用高视觉Token预算(1120)，打标/描述用中低预算。\n"
        "MiniMax H3 任务: 图生视频用图像1作首帧；首尾帧视频需同时接图像1(首帧)和图像2(尾帧)，"
        "自定义问题可补充具体需求。\n"
        "H3 任务输出较长，建议把\"最大生成长度\"选到 4096。"
    )

    def analyze(self, 图像1, 任务类型, 自定义问题, 模型名称, 量化方式, 视觉Token预算,
                最大生成长度, seed, 思考模式, 运行后立即卸载, 自动下载模型, 下载源, 图像2=None):
        # 1. 确定提示词
        prompt = TASK_PROMPTS.get(任务类型)
        if prompt is None:
            prompt = 自定义问题.strip()
            if not prompt:
                raise ValueError("任务类型为\"自定义问题\"时，请在\"自定义问题\"中输入内容")
        elif 任务类型.startswith("MiniMax H3") and 自定义问题.strip():
            # H3 预设任务的提示词承诺参考用户需求，需把自定义问题一并传入
            prompt += f"\n\n用户附加要求: {自定义问题.strip()}"
        if 任务类型 == "MiniMax H3 | 首尾帧视频" and 图像2 is None:
            raise ValueError("任务类型「MiniMax H3 | 首尾帧视频」需要同时连接「图像1」和「图像2」(图像2作尾帧)")

        # 2. 图像存为临时文件传给子进程 (取批次第一张)
        tmp_path = _save_temp_image(图像1[0])
        tmp_files = [tmp_path]

        payload = {
            "quant": 量化方式,
            "prompt": prompt,
            "enable_thinking": 思考模式,
            "visual_token_budget": int(视觉Token预算),
            "max_new_tokens": int(最大生成长度),
            "seed": seed,
        }
        if 图像2 is not None:
            # 两张图同接: 改用带标签的 media 列表，自定义问题可分别引用【图像1】【图像2】
            tmp_path2 = _save_temp_image(图像2[0])
            tmp_files.append(tmp_path2)
            payload["media"] = [
                {"kind": "image", "label": "图像1", "paths": [tmp_path]},
                {"kind": "image", "label": "图像2", "paths": [tmp_path2]},
            ]
        else:
            payload["image_path"] = tmp_path

        # 3. 子进程推理
        try:
            return _run_gemma(模型名称, 自动下载模型, 下载源, payload, unload_after=运行后立即卸载)
        finally:
            _cleanup_files(tmp_files)


class Gemma_Chat_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_gemma_models()
        all_models = sorted(set(installed + [DEFAULT_GEMMA_MODEL]))
        prompt_keys = list(GEMMA_SYSTEM_PROMPTS.keys())
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": "你好，请介绍一下你自己。", "placeholder": "在此输入对话内容..."}),
                "系统指令类型": (prompt_keys, {"default": prompt_keys[0]}),
                "模型名称": (all_models, {"default": DEFAULT_GEMMA_MODEL}),
                "量化方式": (["4bit", "8bit", "bf16"], {"default": "4bit"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "思考模式": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HF Mirror", "HuggingFace"], {"default": "ModelScope"}),
                "最大生成长度": (GEMMA_NEW_TOKEN_OPTIONS, {"default": "2048"}),
            },
            "optional": {
                # 编号动态输入: 前端默认每类只显示 1 个槽位，连接后自动追加下一个 (上限见 CHAT_MAX_*)；
                # 子进程给每个媒体插入【图像1】等标签，提示词可按编号精确引用
                **{f"图像{i}": ("IMAGE",) for i in range(1, CHAT_MAX_IMAGE + 1)},
                **{f"视频{i}": ("IMAGE",) for i in range(1, CHAT_MAX_VIDEO + 1)},
                **{f"音频{i}": ("AUDIO",) for i in range(1, CHAT_MAX_AUDIO + 1)},
                "视频分析": (FRAME_SAMPLE_MODES, {"default": "快速"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("回复内容", "思考过程")
    OUTPUT_NODE = True
    FUNCTION = "chat"
    CATEGORY = "💬 AI人工智能/谷歌系列"
    DESCRIPTION = (
        "基于 Gemma 4 的多模态对话节点 (隔离进程运行)。\n"
        "可同时接入多个图像(最多9张)/视频(最多3段)/音频(最多3段，各≤30秒)，都不接入时为纯文本对话。\n"
        "每类输入默认只显示 1 个，连接后自动追加下一个空槽位 。\n"
        "每个媒体会标注【图像1】【视频1】【音频1】等编号，提示词里可直接引用对应编号。\n"
        "视频接\"加载视频\"的图像序列输出，\"视频分析\"三档: 快速(8帧)/中等(30帧)/详细(全部帧)。支持思考模式。\n"
        "MiniMax H3 系统指令: 文生视频按提示词需求直接生成；参考生视频需先接入参考素材(图像/视频/音频)，"
        "否则无素材可分析。\n"
        "H3 指令输出较长，建议把\"最大生成长度\"选到 4096。"
    )

    def chat(self, 提示词, 系统指令类型, 模型名称, 量化方式, 最大生成长度, seed,
             思考模式, 运行后立即卸载, 自动下载模型, 下载源, 视频分析="快速", **kwargs):
        if not 提示词.strip():
            raise ValueError("请输入提示词")

        tmp_files = []
        media = []
        payload = {
            "quant": 量化方式,
            "prompt": 提示词,
            "system": GEMMA_SYSTEM_PROMPTS.get(系统指令类型, "You are a helpful assistant."),
            "enable_thinking": 思考模式,
            "max_new_tokens": int(最大生成长度),
            "seed": seed,
        }
        # 按 图像 -> 视频 -> 音频 的固定顺序收集已接入的媒体，编号与插槽名一致
        for i in range(1, CHAT_MAX_IMAGE + 1):
            img = kwargs.get(f"图像{i}")
            if img is not None:
                p = _save_temp_image(img[0])
                tmp_files.append(p)
                media.append({"kind": "image", "label": f"图像{i}", "paths": [p]})
        for i in range(1, CHAT_MAX_VIDEO + 1):
            vid = kwargs.get(f"视频{i}")
            if vid is not None:
                indices = _frame_sample_indices(vid.shape[0], 视频分析)
                paths = []
                for idx in indices:
                    p = _save_temp_image(vid[idx])
                    tmp_files.append(p)
                    paths.append(p)
                media.append({"kind": "video", "label": f"视频{i}", "paths": paths})
        for i in range(1, CHAT_MAX_AUDIO + 1):
            aud = kwargs.get(f"音频{i}")
            if aud is not None:
                p = _save_temp_audio(aud)
                tmp_files.append(p)
                media.append({"kind": "audio", "label": f"音频{i}", "paths": [p]})
        if media:
            payload["media"] = media

        try:
            return _run_gemma(模型名称, 自动下载模型, 下载源, payload, unload_after=运行后立即卸载)
        finally:
            _cleanup_files(tmp_files)


# 音频理解任务预设 (显示文本 -> 提示词, None 表示使用自定义问题)
AUDIO_TASK_PROMPTS = {
    "转写文字 | 原样转写语音内容": (
        "Transcribe the following speech segment in its original language. "
        "Only output the transcription, with no extra explanations. "
        "When transcribing numbers, write the digits (e.g. 1.7 instead of one point seven)."
    ),
    "翻译中文 | 语音内容翻译成中文": "请听这段音频，并将其中的语音内容翻译成简体中文。只输出翻译结果。",
    "内容描述 | 描述音频中的声音内容": "请用中文描述这段音频的内容，包括语音、音乐、环境音等信息。",
    "自定义问题 | 使用下方自定义问题提问": None,
}


class Gemma_Audio_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_gemma_models()
        all_models = sorted(set(installed + [DEFAULT_GEMMA_MODEL]))
        task_keys = list(AUDIO_TASK_PROMPTS.keys())
        return {
            "required": {
                "音频": ("AUDIO",),
                "任务类型": (task_keys, {"default": task_keys[0]}),
                "自定义问题": ("STRING", {"multiline": True, "default": "", "placeholder": "任务类型选\"自定义问题\"时在此输入..."}),
                "模型名称": (all_models, {"default": DEFAULT_GEMMA_MODEL}),
                "量化方式": (["4bit", "8bit", "bf16"], {"default": "4bit"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HF Mirror", "HuggingFace"], {"default": "ModelScope"}),
                "最大生成长度": (GEMMA_NEW_TOKEN_OPTIONS, {"default": "1024"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回复内容",)
    OUTPUT_NODE = True
    FUNCTION = "analyze"
    CATEGORY = "💬 AI人工智能/谷歌系列"
    DESCRIPTION = (
        "基于 Gemma 4 的音频理解节点 (隔离进程运行)。\n"
        "支持语音转写、翻译成中文、音频内容描述和自定义提问，可直接连接\"加载音频\"节点。\n"
        "注意: 模型不支持中文语音 (官方音频基准注明 Excluding Chinese)，"
        "中文语音请使用 Qwen 语音识别 (ASR) 节点；音频最长 30 秒。"
    )

    def analyze(self, 音频, 任务类型, 自定义问题, 模型名称, 量化方式, 最大生成长度,
                seed, 运行后立即卸载, 自动下载模型, 下载源):
        prompt = AUDIO_TASK_PROMPTS.get(任务类型)
        if prompt is None:
            prompt = 自定义问题.strip()
            if not prompt:
                raise ValueError("任务类型为\"自定义问题\"时，请在\"自定义问题\"中输入内容")

        tmp_path = _save_temp_audio(音频)
        try:
            content, _ = _run_gemma(
                模型名称, 自动下载模型, 下载源,
                {
                    "quant": 量化方式,
                    "audio_path": tmp_path,
                    "prompt": prompt,
                    "max_new_tokens": int(最大生成长度),
                    "seed": seed,
                },
                unload_after=运行后立即卸载,
            )
            return (content,)
        finally:
            _cleanup_files([tmp_path])


# 视频理解任务预设 (显示文本 -> 提示词, None 表示使用自定义问题)
VIDEO_TASK_PROMPTS = {
    "内容描述 | 中文描述视频内容": (
        "以下是同一段视频按时间顺序抽取的帧。请用中文描述这段视频的内容，"
        "包括主体、动作、场景变化和整体氛围。"
    ),
    "剧情摘要 | 总结视频剧情要点": (
        "以下是同一段视频按时间顺序抽取的帧。请用中文总结这段视频的剧情要点，按时间顺序列出。"
    ),
    "MiniMax H3 | 做同款视频": (
        "以下是同一段参考视频按时间顺序抽取的帧。你是MiniMax H3视频模型的提示词专家，"
        "请全面分析这些帧，总结其主体、场景、剧情、镜头语言（景别、运镜、剪辑节奏）、"
        "色调风格、光影与声音氛围，然后根据用户在自定义问题中的需求撰写一份“同款”"
        "MiniMax H3文生视频高质量提示词。\n"
        "MiniMax H3提示词规则：\n"
        "1. 采用四段式结构：【参考素材说明】【核心创意】【画面过程描述】【整体要求补充】。\n"
        "2. 【参考素材说明】写：无参考素材（纯文字生成视频），风格复刻自一部参考作品。\n"
        "3. 【核心创意】写明“N秒视频”（N取4~15），用一句话锁定主体、地点、事件、题材风格，概括与原视频同款主题与看点。\n"
        "4. 【画面过程描述】复刻原视频的分镜节奏，按时间点拆分镜，每段写“正向：… - 反向：不要…”，写明景别与运镜；"
        "台词长短须与镜头长短对齐，画面中出现的文字、Logo须给出原文；"
        "若用户需求中包含具体台词/对白/歌词，其内容必须一字不差地完整保留在输出提示词中并安排到对应镜头，"
        "不得省略、缩写、改写或概括。\n"
        "5. 【整体要求补充】包含：▍影像风格（复刻原视频的色调、胶片质感、布光）；"
        "▍声音设计（H3输出自带原生立体声：复刻原视频的音乐风格、环境音效、台词语气；原视频无配乐则写“非叙事性音乐：N/A”）；"
        "▍附加要求（转场方式、禁止元素如“不要字幕、不要水印”、一致性要求）。\n"
        "6. 描述尽量具体、直接、可视化，少用需要意会的比喻。\n"
        "7. 全文不超过7000字符，直接输出完整提示词，不要输出分析过程。\n"
        "若用户在自定义问题中附加了具体要求，请优先满足其要求。"
    ),
    "自定义问题 | 使用下方自定义问题提问": None,
}


class Gemma_Video_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_gemma_models()
        all_models = sorted(set(installed + [DEFAULT_GEMMA_MODEL]))
        task_keys = list(VIDEO_TASK_PROMPTS.keys())
        return {
            "required": {
                "图像序列": ("IMAGE",),
                "任务类型": (task_keys, {"default": task_keys[0]}),
                "自定义问题": ("STRING", {"multiline": True, "default": "", "placeholder": "任务类型选\"自定义问题\"时在此输入..."}),
                "视频分析": (FRAME_SAMPLE_MODES, {"default": "快速"}),
                "模型名称": (all_models, {"default": DEFAULT_GEMMA_MODEL}),
                "量化方式": (["4bit", "8bit", "bf16"], {"default": "4bit"}),
                "视觉Token预算": (VISUAL_TOKEN_BUDGETS, {"default": "140"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HF Mirror", "HuggingFace"], {"default": "ModelScope"}),
                "最大生成长度": (GEMMA_NEW_TOKEN_OPTIONS, {"default": "1024"}),
            },
            "optional": {
                "音频": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回复内容",)
    OUTPUT_NODE = True
    FUNCTION = "analyze"
    CATEGORY = "💬 AI人工智能/谷歌系列"
    DESCRIPTION = (
        "基于 Gemma 4 的视频理解节点 (隔离进程运行)。\n"
        "接\"加载视频\"的图像序列输出，按\"视频分析\"档位抽帧后理解视频内容，可选同时接入音频。\n"
        "视频分析三档: 快速(8帧)/中等(30帧)/详细(抽取全部帧)，抽帧多时建议用低视觉Token预算(70/140)。\n"
        "注意: 画面理解不受语言限制，但可选音频输入不支持中文语音 "
        "(官方音频基准注明 Excluding Chinese)，中文音轨请断开音频或改用 Qwen 语音识别 (ASR) 节点。\n"
        "MiniMax H3 做同款视频: 分析参考视频后生成同款文生视频提示词，自定义问题可补充具体需求。\n"
        "输出较长，建议把\"最大生成长度\"选到 4096。"
    )

    def analyze(self, 图像序列, 任务类型, 自定义问题, 视频分析, 模型名称, 量化方式,
                视觉Token预算, 最大生成长度, seed, 运行后立即卸载, 自动下载模型, 下载源, 音频=None):
        prompt = VIDEO_TASK_PROMPTS.get(任务类型)
        if prompt is None:
            prompt = 自定义问题.strip()
            if not prompt:
                raise ValueError("任务类型为\"自定义问题\"时，请在\"自定义问题\"中输入内容")
        elif 任务类型.startswith("MiniMax H3") and 自定义问题.strip():
            # H3 预设任务的提示词承诺参考用户需求，需把自定义问题一并传入
            prompt += f"\n\n用户附加要求: {自定义问题.strip()}"

        # 按档位抽帧 (详细档抽取全部帧)
        indices = _frame_sample_indices(图像序列.shape[0], 视频分析)

        tmp_files = [_save_temp_image(图像序列[i]) for i in indices]
        payload = {
            "quant": 量化方式,
            "image_paths": list(tmp_files),  # 传副本，避免后续追加音频路径污染图像列表
            "prompt": prompt,
            "visual_token_budget": int(视觉Token预算),
            "max_new_tokens": int(最大生成长度),
            "seed": seed,
        }
        if 音频 is not None:
            payload["audio_path"] = _save_temp_audio(音频)
            tmp_files.append(payload["audio_path"])

        try:
            content, _ = _run_gemma(模型名称, 自动下载模型, 下载源, payload, unload_after=运行后立即卸载)
            return (content,)
        finally:
            _cleanup_files(tmp_files)


class Gemma_H3_Template_Node:
    @classmethod
    def INPUT_TYPES(cls):
        installed = get_installed_gemma_models()
        all_models = sorted(set(installed + [DEFAULT_GEMMA_MODEL]))
        template_keys = list(H3_PROMPT_TEMPLATES.keys())
        return {
            "required": {
                "提示词模版": (template_keys, {"default": template_keys[0]}),
                "具体要求": ("STRING", {"multiline": True, "default": "", "placeholder": "留空直接输出模版原文；填写后由 Gemma 优化模版..."}),
                "模型名称": (all_models, {"default": DEFAULT_GEMMA_MODEL}),
                "量化方式": (["4bit", "8bit", "bf16"], {"default": "4bit"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "思考模式": ("BOOLEAN", {"default": False}),
                "运行后立即卸载": ("BOOLEAN", {"default": True}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HF Mirror", "HuggingFace"], {"default": "ModelScope"}),
                "最大生成长度": (GEMMA_NEW_TOKEN_OPTIONS, {"default": "4096"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("提示词", "思考过程")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "💬 AI人工智能/谷歌系列"
    DESCRIPTION = (
        "内置 MiniMax H3 官方手册案例提示词模版，覆盖品牌大片、MG动画、短剧、电商、游戏UI、动画风格化、精准编辑等场景大类。\n"
        "选择模版并在\"具体要求\"填写需求后，由 Gemma 按 H3 官方规则优化模版输出完整提示词；"
        "\"具体要求\"留空时直接输出模版原文，不运行模型。\n"
        "模版中的素材编号(@图片N等)对应实际生成时上传的素材编号，纯文生场景模型会自动改为无参考素材。\n"
        "输出较长，建议把\"最大生成长度\"选到 4096。"
    )

    def generate(self, 提示词模版, 具体要求, 模型名称, 量化方式, seed,
                 思考模式, 运行后立即卸载, 自动下载模型, 下载源, 最大生成长度):
        template = H3_PROMPT_TEMPLATES[提示词模版]
        requirement = 具体要求.strip()
        if not requirement:
            # 无具体要求: 直接输出模版原文，不启动模型
            return (template, "")
        payload = {
            "quant": 量化方式,
            "prompt": f"提示词模版:\n{template}\n\n用户具体要求:\n{requirement}",
            "system": H3_TEMPLATE_OPTIMIZE_PROMPT,
            "enable_thinking": 思考模式,
            "max_new_tokens": int(最大生成长度),
            "seed": seed,
        }
        return _run_gemma(模型名称, 自动下载模型, 下载源, payload, unload_after=运行后立即卸载)
