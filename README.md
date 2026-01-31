# 💬 ComfyUI-Artificial-Intelligence (本地大模型智能助手)

**将强大的本地大语言模型（LLM）无缝集成到您的 ComfyUI 工作流中。无需 API Key，保护隐私，开箱即用。**

## ✨ 核心优势

* **🚀 零门槛，自动部署**：内置国内镜像加速下载，只需选择模型，系统自动拉取。
* **🔒 100% 本地运行**：完全在本地显卡运行，保护隐私，无 API 费用。
* **🇨🇳 全中文友好界面**：参数、选项全汉化，专为中文用户优化。
* **🧠 支持前沿模型**：支持 **Qwen 2.5** 全系列及最新的 **Qwen 3 (4B)**。
* **🎛️ 精细控制**：支持随机种子、温度、Top_P 等专业参数调节。

---

## 💻 安装方法 (Installation)

### 方式一：通过 GitHub 安装 (推荐)

1. 打开您的 ComfyUI 目录，进入 `custom_nodes` 文件夹。
2. 在地址栏输入 `cmd` 或右键打开终端/命令行。
3. 输入以下命令克隆本项目：
```bash
git clone https://github.com/a63976659/ComfyUI-Artificial-Intelligence.git

```


4. 进入插件文件夹并安装依赖：
```bash
cd 您的仓库名
pip install -r requirements.txt

```


5. 重启 ComfyUI。

---

## 📥 模型下载与管理

本插件的模型存放目录为：`ComfyUI/models/LLM`

### 方法 A：自动下载 (最简单)

1. 在节点界面中，将 **"自动下载模型"** 开关设置为 `True`。
2. 在 **"模型名称"** 列表中选择您想要的模型（例如 `Qwen2.5-7B-Instruct`）。
3. 运行工作流（Queue Prompt）。
4. 插件会自动检测本地是否存在该模型，若不存在，将通过国内镜像源自动下载并保存到 `models/LLM` 目录中。
* *注意：首次下载可能需要一定时间，请关注控制台（Console）进度。*



### 方法 B：手动下载 (适用于网络受限环境)

如果您希望手动管理模型，请按照以下步骤操作：

1. **访问模型库**：前往 HuggingFace 或 ModelScope 下载模型文件。
* [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
* [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
* [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) *(注：请查找对应的Qwen仓库)*


2. **创建文件夹**：
在 `ComfyUI/models/LLM` 目录下创建一个与模型名称完全一致的文件夹。例如：
* `ComfyUI/models/LLM/Qwen2.5-7B-Instruct`


3. **放入文件**：
将下载的所有模型文件（`.safetensors`, `config.json`, `tokenizer.json` 等）放入上述文件夹中。
4. **目录结构示例**：
确保您的文件结构如下所示：
```text
ComfyUI/
├── models/
│   ├── LLM/
│   │   ├── Qwen2.5-7B-Instruct/
│   │   │   ├── model-00001-of-00004.safetensors
│   │   │   ├── config.json
│   │   │   ├── tokenizer.json
│   │   │   └── ... (其他文件)

```


5. **刷新 ComfyUI**：下载完成后， ComfyUI 界面,点击键盘刷新按钮的 "R" ，或重启 ComfyUI。

---

## 🧩 节点功能介绍

该插件位于节点菜单的 **Category: 💬 AI人工智能** 下。

### 1. 🧠 LLM 智能翻译

* **用途**：提示词润色、多语言互译（中/英/日/韩等）。
* **特点**：比翻译软件更懂 Stable Diffusion 的语境。

### 2. 💬 LLM 智能对话

* **用途**：创意文案生成、角色扮演、逻辑推理。
* **特点**：支持 System Prompt（人设指令）和 Seed（随机种子）锁定，方便复现结果。

---

## ⚠️ 常见问题

* **报错 `Out of Memory**`：请尝试切换更小的模型（如 1.5B 或 3B 版本），或减小 `最大生成长度`。
* **下载失败**：请检查网络连接，或尝试使用“方法 B”手动下载模型。
