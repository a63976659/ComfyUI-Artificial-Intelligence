# 💬 ComfyUI-Artificial-Intelligence (本地大模型智能助手)

**将强大的本地大语言模型（LLM）无缝集成到您的 ComfyUI 工作流中。无需 API Key，保护隐私，开箱即用。**

## ✨ 核心优势

* **🚀 零门槛，自动部署**：内置国内镜像加速下载，只需选择模型，系统自动拉取。
* **🔒 100% 本地运行**：完全在本地显卡运行，保护隐私，无 API 费用。
* **🇨🇳 全中文友好界面**：参数、选项全汉化，专为中文用户优化。
* **🧠 支持前沿模型**：支持 **Qwen 2.5** 全系列及最新的 **Qwen 3 (4B)**。
* **🎛️ 精细控制**：支持随机种子、温度、Top_P 等专业参数调节。

---

## 更新介绍
## 20260201
* 插件初次创建，主要节点，🧠 LLM 智能翻译 (Qwen)，💬 LLM 智能对话 (Qwen)


## 20260203
* 🔊 Qwen 语音合成 (CustomVoice)用于文生音频，支持9种人声，多国语言，支持 [pause:0.5] 停顿语法。
* 🔊 Qwen 语音设计 (VoiceDesign)用于文生音频，支持通过“声音设计描述”用自然语言定义声音。
* 🔊 Qwen 语音克隆 (VoiceClone)用于声音模仿，给它一段 5-10 秒的某人录音，它就能用那个人的声音说出任何话。
* 🧠 LLM 智能翻译 (Qwen)增加提示词润色功能。



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

### 3. 🔊 Qwen 语音合成 (CustomVoice)

* **用途**：基于官方微调的 9 位高质量预设角色（如 Vivian、Uncle_Fu）进行语音生成，适合稳定商用。
* **特点**：**稳定性极高**，支持**情感指令控制**（如输入“用愤怒咆哮的语气”），支持多语言混合生成。

### 4. 🔊 Qwen 语音设计 (VoiceDesign)

* **用途**：无需参考音频，通过自然语言描述（Prompt）凭空创造独一无二的声音。
* **特点**：真正的**“文本捏音”**，可自定义性别、年龄、音色和语气（如描述“一个稚嫩的五岁女孩，声音尖细且在撒娇”）。

### 5. 🔊 Qwen 语音克隆 (VoiceClone)

* **用途**：输入 5-10 秒参考音频，即可零样本复刻任意人物的声音。
* **特点**：支持**标准模式**（高还原度，需输入参考文本）和**极速模式**（仅提取音色），支持**批次输出**和精细化声学参数控制。
---

## ⚠️ 常见问题

* **报错 `Out of Memory**`：请尝试切换更小的模型（如 1.5B 或 3B 版本），或减小 `最大生成长度`。
* **下载失败**：请检查网络连接，或尝试使用“方法 B”手动下载模型。


---
### **支持开发者**
如果这个插件帮你节省了大量时间，欢迎：
- ⭐ **Star 项目**让更多人看到
- 🐛 **提交 Issue** 帮助改进
- 📖 **分享教程** 帮助其他用户
- 💬 **讨论社区**：[QQ群202018000] 
- ☕ **请作者喝咖啡**
**🔔 关注更新，获取最新功能和模型数据库**
---
**作者**：a63976659  
- [小红书：猪的飞行梦] 
- [哔哩哔哩主页：https://space.bilibili.com/2114638644]

**致谢**：感谢所有测试用户和贡献者！
如果你觉得插件还不错可以点个收藏。


请作者喝奶茶可以扫个码😀😀😀
养家版二维码❥(^_-)
![收款二维码](https://github.com/user-attachments/assets/6394dcb5-4ef4-4fc7-8fc9-9fc98625ce34)

