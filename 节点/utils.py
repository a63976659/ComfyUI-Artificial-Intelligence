import os
import json
import folder_paths
from huggingface_hub import snapshot_download as hf_snapshot_download

# --- 尝试导入 ModelScope ---
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False

# ================= 配置与路径管理 =================

LLM_MODELS_DIR = os.path.join(folder_paths.models_dir, "LLM")
if not os.path.exists(LLM_MODELS_DIR):
    os.makedirs(LLM_MODELS_DIR)

TTS_MODELS_DIR = os.path.join(folder_paths.models_dir, "TTS")
if not os.path.exists(TTS_MODELS_DIR):
    os.makedirs(TTS_MODELS_DIR)

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")

def get_installed_models():
    """扫描本地 LLM 模型 (排除 Gemma 系列, 其由 Gemma 节点单独管理)"""
    if not os.path.exists(LLM_MODELS_DIR):
        return []
    models = [d for d in os.listdir(LLM_MODELS_DIR)
              if os.path.isdir(os.path.join(LLM_MODELS_DIR, d)) and "gemma" not in d.lower()]
    return sorted(models)

def get_installed_tts_models():
    """扫描本地 TTS 模型"""
    if not os.path.exists(TTS_MODELS_DIR):
        return []
    models = [d for d in os.listdir(TTS_MODELS_DIR) if os.path.isdir(os.path.join(TTS_MODELS_DIR, d))]
    return sorted(models)

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"last_model": "Qwen2.5-7B-Instruct"}

def save_config(model_name):
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            old_config = load_config()
            old_config["last_model"] = model_name
            json.dump(old_config, f)
    except Exception as e:
        print(f"[LLM] Config save failed: {e}")

def resolve_llm_model(model_name, auto_download=False):
    """定位本地 LLM 模型目录，不存在时可从 HF Mirror 自动下载 (仅路径，不加载模型)"""
    possible_paths = [
        os.path.join(LLM_MODELS_DIR, model_name),
        os.path.join(LLM_MODELS_DIR, model_name.split("/")[-1]) if "/" in model_name else os.path.join(LLM_MODELS_DIR, model_name)
    ]

    for p in possible_paths:
        if os.path.exists(p) and any(f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(p)):
            return p

    if auto_download:
        print(f"\n[LLM] Model not found locally. Downloading from HF Mirror: {model_name}")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        try:
            target_folder_name = model_name.split("/")[-1] if "/" in model_name else model_name
            download_path = os.path.join(LLM_MODELS_DIR, target_folder_name)
            hf_snapshot_download(repo_id=model_name, local_dir=download_path, resume_download=True, max_workers=4)
            return download_path
        except Exception as e:
            raise Exception(f"Download failed: {e}")
    else:
        raise FileNotFoundError(f"Model {model_name} not found and auto_download is False.")

def resolve_tts_model(model_name, auto_download=False, source="ModelScope"):
    """定位本地 TTS 模型目录，不存在时按下载源自动下载 (仅路径，不加载模型)"""
    target_folder_name = model_name.split("/")[-1] if "/" in model_name else model_name
    possible_paths = [
        os.path.join(TTS_MODELS_DIR, model_name),
        os.path.join(TTS_MODELS_DIR, target_folder_name)
    ]

    for p in possible_paths:
        if os.path.exists(p) and any(f.endswith(".safetensors") for f in os.listdir(p)):
            return p

    if not auto_download:
        raise FileNotFoundError(f"TTS Model {model_name} not found and auto_download is False.")

    download_path = os.path.join(TTS_MODELS_DIR, target_folder_name)
    repo_id = model_name if "/" in model_name else f"Qwen/{model_name}"

    if source == "ModelScope":
        if not HAS_MODELSCOPE:
            raise ImportError("请先安装 modelscope: pip install modelscope")
        print(f"\n[TTS] Downloading from ModelScope: {repo_id} -> {download_path}")
        try:
            ms_snapshot_download(model_id=repo_id, local_dir=download_path)
        except Exception as e:
            raise Exception(f"ModelScope download failed: {e}")

    elif source == "HF Mirror":
        print(f"\n[TTS] Downloading from HF Mirror: {repo_id} -> {download_path}")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        try:
            hf_snapshot_download(repo_id=repo_id, local_dir=download_path, resume_download=True, max_workers=4)
        except Exception as e:
            raise Exception(f"HF Mirror download failed: {e}")

    else: # HuggingFace
        print(f"\n[TTS] Downloading from HuggingFace (Official): {repo_id} -> {download_path}")
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        try:
            hf_snapshot_download(repo_id=repo_id, local_dir=download_path, resume_download=True, max_workers=4)
        except Exception as e:
            raise Exception(f"HuggingFace download failed: {e}")

    return download_path