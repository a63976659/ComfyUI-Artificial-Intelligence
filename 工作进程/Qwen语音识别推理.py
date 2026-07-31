# -*- coding: utf-8 -*-
"""
Qwen3-ASR 隔离进程推理脚本
运行环境: asr_env (transformers==4.57.6 + qwen-asr)，不要在 ComfyUI 主进程中导入本文件。
音频以临时 .wav 文件路径传入 (请求中的 audio_path)，时间戳在本进程内合并为普通字典返回。
"""
import functools

from 协议 import log, main_loop

# 全局模型缓存 (进程内只驻留一份，缓存键含 aligner 状态)
_MODELS = {}


# ================= 兼容性补丁 (qwen-asr 0.0.6 与新版 transformers) =================

def _patch_check_model_inputs():
    """将 transformers 的 check_model_inputs 替换为完全透明的装饰器

    qwen-asr 中使用 @check_model_inputs (不带括号) 或 @check_model_inputs()，
    但新版 transformers 中它是装饰器工厂，且内部 wrapper 可能过滤参数。
    """
    try:
        import transformers.utils.generic as _tg

        def _transparent_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        def _compat_wrapper(func=None, *args, **kwargs):
            if func is not None and callable(func):
                return _transparent_decorator(func)
            return _transparent_decorator

        _tg.check_model_inputs = _compat_wrapper
    except (ImportError, AttributeError):
        pass


def _patch_rope_init_functions():
    """修复 ROPE_INIT_FUNCTIONS 缺少 'default' 键的问题

    qwen-asr 的模型配置中 rope_type 为 'default' (原始 RoPE 实现，无缩放)，
    部分 transformers 版本的 ROPE_INIT_FUNCTIONS 字典中未包含该键。
    """
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if 'default' not in ROPE_INIT_FUNCTIONS:
            import torch

            def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
                config.standardize_rope_params()
                rope_parameters_dict = (
                    config.rope_parameters[layer_type]
                    if layer_type is not None
                    else config.rope_parameters
                )
                base = rope_parameters_dict["rope_theta"]
                partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
                head_dim = (
                    getattr(config, "head_dim", None)
                    or config.hidden_size // config.num_attention_heads
                )
                dim = int(head_dim * partial_rotary_factor)
                inv_freq = 1.0 / (
                    base ** (
                        torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim
                    )
                )
                return inv_freq, 1.0

            ROPE_INIT_FUNCTIONS['default'] = _compute_default_rope_parameters
    except (ImportError, AttributeError):
        pass


def _patch_qwen3_asr_config():
    """修复 Qwen3ASRConfig / Qwen3ASRThinkerConfig 子配置初始化顺序

    新版 transformers 的严格校验在 super().__init__() 中就会访问 thinker_config，
    而原始代码在 super().__init__() 之后才赋值，导致 AttributeError。
    """
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
            Qwen3ASRConfig as _Cfg,
            Qwen3ASRThinkerConfig as _ThinkerCfg,
            Qwen3ASRAudioEncoderConfig as _AudioCfg,
            Qwen3ASRTextConfig as _TextCfg,
        )
        try:
            from transformers import PreTrainedConfig
        except ImportError:
            from transformers import PretrainedConfig as PreTrainedConfig

        def _patched_cfg_init(self, thinker_config=None, support_languages=None, **kwargs):
            if thinker_config is None:
                thinker_config = {}
            if isinstance(thinker_config, dict):
                self.thinker_config = _ThinkerCfg(**thinker_config)
            else:
                self.thinker_config = thinker_config
            self.support_languages = support_languages
            PreTrainedConfig.__init__(self, **kwargs)

        _Cfg.__init__ = _patched_cfg_init

        def _patched_thinker_init(self, audio_config=None, text_config=None,
                                  audio_token_id=151646, audio_start_token_id=151647,
                                  user_token_id=872, initializer_range=0.02, **kwargs):
            if isinstance(audio_config, dict):
                self.audio_config = _AudioCfg(**audio_config)
            elif audio_config is None:
                self.audio_config = _AudioCfg()
            else:
                self.audio_config = audio_config

            if isinstance(text_config, dict):
                self.text_config = _TextCfg(**text_config)
            elif text_config is None:
                self.text_config = _TextCfg()
            else:
                self.text_config = text_config

            self.audio_token_id = audio_token_id
            self.audio_start_token_id = audio_start_token_id
            self.user_token_id = user_token_id
            self.initializer_range = initializer_range
            PreTrainedConfig.__init__(self, **kwargs)

        _ThinkerCfg.__init__ = _patched_thinker_init
    except (ImportError, AttributeError) as e:
        log(f"警告: Qwen3ASRConfig 补丁未生效: {e}")


def _patch_rotary_embedding_init():
    """修复 Qwen3ASRThinkerTextRotaryEmbedding 中 rope_scaling 为 None 时的崩溃"""
    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRThinkerTextRotaryEmbedding as _RotaryEmbed,
        )
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        def _patched_rotary_init(self, config, device=None):
            # 不能用无参 super()，必须通过 type(self) 定位正确的父类
            super(type(self), self).__init__()
            if getattr(config, "rope_scaling", None) is not None:
                self.rope_type = config.rope_scaling.get("rope_type", "default")
                self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])
            else:
                self.rope_type = "default"
                self.mrope_section = [24, 20, 20]

            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
            self.config = config
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.original_inv_freq = self.inv_freq

        _RotaryEmbed.__init__ = _patched_rotary_init

        # 部分 transformers 版本的 _init_weights 在 rope_type='default' 时
        # 会访问 module.compute_default_rope_parameters
        if not hasattr(_RotaryEmbed, 'compute_default_rope_parameters'):
            _RotaryEmbed.compute_default_rope_parameters = property(lambda self: self.rope_init_fn)
    except (ImportError, AttributeError) as e:
        log(f"警告: RotaryEmbedding 补丁未生效: {e}")


def _patch_thinker_for_cg_init():
    """修复 Qwen3ASRThinkerForConditionalGeneration.__init__ 中的属性访问问题

    原始代码直接访问 config.pad_token_id / config.classify_num，
    新版 transformers 严格检查下不存在的属性会抛 AttributeError。
    """
    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRThinkerForConditionalGeneration as _ThinkerCG,
            Qwen3ASRAudioEncoder,
            Qwen3ASRThinkerTextModel,
        )
        import torch.nn as nn

        def _patched_thinker_cg_init(self, config):
            super(type(self), self).__init__(config)

            self.audio_tower = Qwen3ASRAudioEncoder._from_config(config.audio_config)
            self.vocab_size = config.text_config.vocab_size
            self.model = Qwen3ASRThinkerTextModel._from_config(config.text_config)

            if "forced_aligner" in config.model_type:
                classify_num = getattr(config, "classify_num", config.text_config.vocab_size)
                self.lm_head = nn.Linear(config.text_config.hidden_size, classify_num, bias=False)
            else:
                self.lm_head = nn.Linear(config.text_config.hidden_size,
                                         config.text_config.vocab_size, bias=False)

            self.pad_token_id = getattr(self.config, "pad_token_id", None)
            if self.pad_token_id is None:
                self.pad_token_id = -1

            self.rope_deltas = None
            self.post_init()

        _ThinkerCG.__init__ = _patched_thinker_cg_init
    except (ImportError, AttributeError) as e:
        log(f"警告: ThinkerForConditionalGeneration 补丁未生效: {e}")


def _apply_patches():
    """按顺序应用全部兼容补丁 (须在导入 qwen_asr 前后配合调用)"""
    _patch_check_model_inputs()
    _patch_rope_init_functions()
    _patch_qwen3_asr_config()
    _patch_rotary_embedding_init()
    _patch_thinker_for_cg_init()


# ================= 模型加载 =================

def load_model(model_path, aligner_path):
    """按路径加载/复用模型，aligner_path 非空时挂载 ForcedAligner (用于时间戳)"""
    import torch

    cache_key = f"{model_path}|{aligner_path or ''}"
    if cache_key in _MODELS:
        return _MODELS[cache_key]

    if _MODELS:
        log("模型配置变化，卸载旧模型...")
        _MODELS.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    _apply_patches()
    from qwen_asr import Qwen3ASRModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    log(f"加载模型: {model_path} -> {device} (时间戳={'是' if aligner_path else '否'})")
    load_kwargs = {"dtype": dtype, "device_map": device}
    if aligner_path:
        log(f"挂载对齐模型: {aligner_path}")
        load_kwargs["forced_aligner"] = aligner_path
        load_kwargs["forced_aligner_kwargs"] = {"dtype": dtype, "device_map": device}

    model = Qwen3ASRModel.from_pretrained(model_path, **load_kwargs)
    _MODELS[cache_key] = model
    log("模型加载完成")
    return model


# ================= 时间戳合并 =================

# 分段规则参数
_GAP_SPLIT_SEC = 0.6        # 相邻两词的静音间隔超过该值时强制断句 (前奏/间奏/句间停顿)
_MAX_SEGMENT_SEC = 8.0      # 单段最长时长，超出后强制断句
_MIN_SEGMENT_SEC = 0.5      # 短于该值的碎片段落尝试与相邻段合并
_TEXT_SEARCH_WINDOW = 8     # 词在完整文本中定位时的前向搜索窗口 (字符)

_PUNCTUATIONS = set('。！？.!?，,；;、\n')


def _normalize_words(time_stamps):
    """将对齐器返回的词级条目整理为 (文本, 起始秒, 结束秒) 列表，丢弃空文本"""
    words = []
    for ts in time_stamps:
        w_text = (getattr(ts, 'text', '') or '').strip()
        if not w_text:
            continue
        words.append((w_text, getattr(ts, 'start_time', None), getattr(ts, 'end_time', None)))
    return words


def _merge_short_segments(segments):
    """合并过短的碎片段落，但不跨越较大静音间隔、不突破单段时长上限"""
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        too_short = (prev["end"] - prev["start"] < _MIN_SEGMENT_SEC
                     or seg["end"] - seg["start"] < _MIN_SEGMENT_SEC)
        gap = seg["start"] - prev["end"]
        if (too_short and gap <= _GAP_SPLIT_SEC
                and seg["end"] - prev["start"] <= _MAX_SEGMENT_SEC):
            prev["end"] = seg["end"]
            prev["text"] = (prev["text"] + seg["text"]).strip()
        else:
            merged.append(seg)
    return merged


def _merge_timestamps_by_text(full_text, time_stamps):
    """将词级时间戳按 标点 / 静音间隔 / 段长上限 三重规则切分为可读段落

    每段起止时间严格取自本段第一个词的 start_time 与最后一个词的 end_time，
    因此段落时间范围不会跨越前奏、间奏等无人声区间 (音乐类音频尤为明显)。
    """
    if not time_stamps or not full_text:
        return []

    words = _normalize_words(time_stamps)
    if not words:
        return []

    segments = []
    text_ptr = 0
    text_len = len(full_text)
    seg_start = None
    seg_end = None
    seg_text = ""

    for i, (w_text, start, end) in enumerate(words):
        if seg_start is None and start is not None:
            seg_start = start
        if end is not None:
            seg_end = end

        # 在带标点的完整输出文本中定位当前词。限定搜索窗口，避免重复词 (歌词副歌)
        # 导致指针大跨度前跳、把整段文本吞入当前段落
        limit = text_ptr + len(w_text) + _TEXT_SEARCH_WINDOW
        found_idx = full_text.find(w_text, text_ptr, limit)
        if found_idx == -1:
            found_idx = full_text.lower().find(w_text.lower(), text_ptr, limit)

        if found_idx != -1:
            seg_text += full_text[text_ptr: found_idx + len(w_text)]
            text_ptr = found_idx + len(w_text)
        else:
            # 分词差异导致窗口内未命中: 按序消费等长文本，避免指针停滞后续整体错位
            take = full_text[text_ptr: text_ptr + len(w_text)]
            seg_text += take or w_text
            text_ptr += len(take)

        # 吸收紧随其后的标点与空格
        has_punctuation = False
        while text_ptr < text_len:
            c = full_text[text_ptr]
            if c in _PUNCTUATIONS:
                has_punctuation = True
            elif c != ' ':
                break
            seg_text += c
            text_ptr += 1

        # 断句判定: 标点 -> 与下一个词之间的静音 -> 段落时长超限
        should_break = has_punctuation
        if not should_break and i + 1 < len(words):
            next_start = words[i + 1][1]
            if next_start is not None and end is not None and next_start - end > _GAP_SPLIT_SEC:
                should_break = True
        if not should_break and seg_start is not None and seg_end is not None:
            if seg_end - seg_start >= _MAX_SEGMENT_SEC:
                should_break = True

        if should_break:
            segments.append({
                "start": round(seg_start if seg_start is not None else 0.0, 3),
                "end": round(seg_end if seg_end is not None else 0.0, 3),
                "text": seg_text.strip(),
            })
            seg_start = None
            seg_end = None
            seg_text = ""

    # 兜底: 处理最后一段以及尾部残余文本
    if seg_text.strip() or seg_start is not None:
        if text_ptr < text_len:
            seg_text += full_text[text_ptr:]
        segments.append({
            "start": round(seg_start if seg_start is not None else 0.0, 3),
            "end": round(seg_end if seg_end is not None else 0.0, 3),
            "text": seg_text.strip(),
        })

    return _merge_short_segments(segments)


# ================= 请求处理 =================

def handle_generate(req):
    """识别单个音频，返回文本 / 语种 / 合并后的时间戳"""
    model = load_model(req["model_path"], req.get("aligner_path"))

    language = req.get("language") or None
    context = req.get("context") or None
    with_timestamps = bool(req.get("return_time_stamps", False))

    kwargs = {
        "audio": [req["audio_path"]],
        "language": [language] if language else None,
        "return_time_stamps": with_timestamps,
    }
    if context:
        kwargs["context"] = [context]

    log(f"识别中... 语言={language or '自动'}")
    result = model.transcribe(**kwargs)[0]
    text = result.text

    timestamps = []
    if with_timestamps:
        raw = getattr(result, "time_stamps", None)
        if raw:
            timestamps = _merge_timestamps_by_text(text, raw)

    log(f"识别完成 (语种={result.language}): {text[:40]}...")
    return {"ok": True, "language": result.language, "text": text, "timestamps": timestamps}


def handle_warmup(req):
    """仅把模型加载进常驻子进程 (不做识别)，供实时翻译会话预热，避免首句才加载"""
    load_model(req["model_path"], req.get("aligner_path"))
    return {"ok": True, "warmed": True}


if __name__ == "__main__":
    main_loop({"generate": handle_generate, "warmup": handle_warmup})
