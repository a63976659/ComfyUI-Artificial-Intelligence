import os
import json
import torch
import folder_paths
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# ================= 配置与路径管理 =================

LLM_MODELS_DIR = os.path.join(folder_paths.models_dir, "LLM")
if not os.path.exists(LLM_MODELS_DIR):
    os.makedirs(LLM_MODELS_DIR)

# 获取当前文件(utils.py)的上一级目录，即插件根目录，以保持config.json位置不变
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")

# 全局模型缓存 (Tokenizer, Model)
LOADED_MODELS = {}

def get_installed_models():
    """扫描本地模型"""
    if not os.path.exists(LLM_MODELS_DIR):
        return []
    models = [d for d in os.listdir(LLM_MODELS_DIR) if os.path.isdir(os.path.join(LLM_MODELS_DIR, d))]
    return sorted(models)

def load_config():
    """读取配置"""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"last_model": "Qwen2.5-7B-Instruct"}

def save_config(model_name):
    """保存配置"""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump({"last_model": model_name}, f)
    except Exception as e:
        print(f"[LLM] Config save failed: {e}")

def load_llm_model(model_name, device, auto_download=False):
    """统一的模型加载函数，支持自动下载"""
    # 1. 确定路径
    possible_paths = [
        os.path.join(LLM_MODELS_DIR, model_name),
        os.path.join(LLM_MODELS_DIR, model_name.split("/")[-1]) if "/" in model_name else os.path.join(LLM_MODELS_DIR, model_name)
    ]
    
    model_path = None
    for p in possible_paths:
        if os.path.exists(p) and any(f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(p)):
            model_path = p
            break
    
    # 2. 下载逻辑
    if not model_path:
        if auto_download:
            print(f"\n[LLM] Model not found locally. Downloading: {model_name}")
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            try:
                target_folder_name = model_name.split("/")[-1] if "/" in model_name else model_name
                download_path = os.path.join(LLM_MODELS_DIR, target_folder_name)
                snapshot_download(repo_id=model_name, local_dir=download_path, resume_download=True, max_workers=4)
                model_path = download_path
                print(f"[LLM] Download completed: {model_path}")
            except Exception as e:
                raise Exception(f"Download failed: {e}")
        else:
            raise FileNotFoundError(f"Model {model_name} not found and auto_download is False.")

    # 3. 加载到显存 (带缓存)
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