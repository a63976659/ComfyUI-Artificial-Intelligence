import os
import json
import torch
import gc
import folder_paths
from huggingface_hub import snapshot_download as hf_snapshot_download

# --- 尝试导入 ModelScope ---
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False

# --- 尝试导入 Qwen-TTS ---
try:
    from qwen_tts import Qwen3TTSModel
    HAS_QWEN_TTS = True
except ImportError:
    HAS_QWEN_TTS = False

# ================= 配置与路径管理 =================

LLM_MODELS_DIR = os.path.join(folder_paths.models_dir, "LLM")
if not os.path.exists(LLM_MODELS_DIR):
    os.makedirs(LLM_MODELS_DIR)

TTS_MODELS_DIR = os.path.join(folder_paths.models_dir, "TTS")
if not os.path.exists(TTS_MODELS_DIR):
    os.makedirs(TTS_MODELS_DIR)

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")

# 全局模型缓存
LOADED_MODELS = {}
LOADED_TTS_MODELS = {} 

def get_installed_models():
    """扫描本地 LLM 模型"""
    if not os.path.exists(LLM_MODELS_DIR):
        return []
    models = [d for d in os.listdir(LLM_MODELS_DIR) if os.path.isdir(os.path.join(LLM_MODELS_DIR, d))]
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

def load_llm_model(model_name, device, auto_download=False):
    # (LLM 逻辑保持不变)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    possible_paths = [
        os.path.join(LLM_MODELS_DIR, model_name),
        os.path.join(LLM_MODELS_DIR, model_name.split("/")[-1]) if "/" in model_name else os.path.join(LLM_MODELS_DIR, model_name)
    ]
    
    model_path = None
    for p in possible_paths:
        if os.path.exists(p) and any(f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(p)):
            model_path = p
            break
    
    if not model_path:
        if auto_download:
            print(f"\n[LLM] Model not found locally. Downloading from HF Mirror: {model_name}")
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            try:
                target_folder_name = model_name.split("/")[-1] if "/" in model_name else model_name
                download_path = os.path.join(LLM_MODELS_DIR, target_folder_name)
                hf_snapshot_download(repo_id=model_name, local_dir=download_path, resume_download=True, max_workers=4)
                model_path = download_path
            except Exception as e:
                raise Exception(f"Download failed: {e}")
        else:
            raise FileNotFoundError(f"Model {model_name} not found and auto_download is False.")

    global LOADED_MODELS
    if model_path not in LOADED_MODELS:
        print(f"[LLM] Loading model from {model_path}...")
        torch.cuda.empty_cache()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype="auto", 
                trust_remote_code=True
            )
            LOADED_MODELS[model_path] = (tokenizer, model)
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    return LOADED_MODELS[model_path]

def load_tts_model_data(model_name, device, auto_download=False, source="ModelScope"):
    """
    加载 TTS 模型
    修复: 
    1. device_map=device (明确指定为 "cuda" 或 "cpu")
    2. 移除 model.to() 调用
    """
    target_folder_name = model_name.split("/")[-1] if "/" in model_name else model_name
    possible_paths = [
        os.path.join(TTS_MODELS_DIR, model_name),
        os.path.join(TTS_MODELS_DIR, target_folder_name)
    ]
    
    model_path = None
    for p in possible_paths:
        if os.path.exists(p) and any(f.endswith(".safetensors") for f in os.listdir(p)):
            model_path = p
            break
    
    if not model_path:
        if auto_download:
            download_path = os.path.join(TTS_MODELS_DIR, target_folder_name)
            repo_id = model_name if "/" in model_name else f"Qwen/{model_name}"
            
            if source == "ModelScope":
                if not HAS_MODELSCOPE:
                    raise ImportError("请先安装 modelscope: pip install modelscope")
                print(f"\n[TTS] Downloading from ModelScope: {repo_id} -> {download_path}")
                try:
                    ms_snapshot_download(model_id=repo_id, local_dir=download_path)
                    model_path = download_path
                except Exception as e:
                    raise Exception(f"ModelScope download failed: {e}")
            
            elif source == "HF Mirror":
                print(f"\n[TTS] Downloading from HF Mirror: {repo_id} -> {download_path}")
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                try:
                    hf_snapshot_download(repo_id=repo_id, local_dir=download_path, resume_download=True, max_workers=4)
                    model_path = download_path
                except Exception as e:
                    raise Exception(f"HF Mirror download failed: {e}")

            else: # HuggingFace
                print(f"\n[TTS] Downloading from HuggingFace (Official): {repo_id} -> {download_path}")
                if "HF_ENDPOINT" in os.environ:
                    del os.environ["HF_ENDPOINT"]
                try:
                    hf_snapshot_download(repo_id=repo_id, local_dir=download_path, resume_download=True, max_workers=4)
                    model_path = download_path
                except Exception as e:
                    raise Exception(f"HuggingFace download failed: {e}")
        else:
            raise FileNotFoundError(f"TTS Model {model_name} not found and auto_download is False.")

    global LOADED_TTS_MODELS
    if model_path not in LOADED_TTS_MODELS:
        # device 是 "cuda" 或 "cpu" 字符串
        print(f"[TTS] Loading model from {model_path} to {device}...")
        torch.cuda.empty_cache()
        
        if not HAS_QWEN_TTS:
            raise ImportError("Critical Dependency Missing: Please run 'pip install qwen-tts'")

        try:
            if "cuda" in str(device):
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            # --- 关键修改 ---
            # 1. 直接将 device_map 设为 "cuda" 或 "cpu"
            # 2. Qwen3TTSModel 是一个 wrapper，没有 .to() 方法，
            #    所以我们依赖 from_pretrained 的 device_map 参数来完成设备移动
            model = Qwen3TTSModel.from_pretrained(
                model_path, 
                device_map=device, # 直接传入设备字符串
                torch_dtype=dtype
            )
            
            # 3. 移除了 model.to(device)
            # model.eval() # wrapper 也不一定有 eval，默认就是推理模式
            
            model.model_type_str = model_name 
            LOADED_TTS_MODELS[model_path] = model
            print("[TTS] Model loaded successfully.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to load TTS model: {e}")
    
    return LOADED_TTS_MODELS[model_path]

def unload_tts_model(model_name):
    """
    强制卸载指定模型并清空显存
    """
    target_folder_name = model_name.split("/")[-1] if "/" in model_name else model_name
    possible_paths = [
        os.path.join(TTS_MODELS_DIR, model_name),
        os.path.join(TTS_MODELS_DIR, target_folder_name)
    ]
    
    model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break
    
    global LOADED_TTS_MODELS
    if model_path and model_path in LOADED_TTS_MODELS:
        print(f"[TTS] Unloading model: {model_name} to free VRAM...")
        del LOADED_TTS_MODELS[model_path]
    
    # 强制执行垃圾回收和显存清理
    gc.collect()
    torch.cuda.empty_cache()