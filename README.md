# MiniMax H3 Semantic Bridge Trainer · T8

**语言 / Language：** [简体中文](README.md) · [English](README.en.md)

可复用的 MiniMax H3 **Semantic Bridge 文本条件训练器**：Windows 图形启动器、命令行训练及动漫战斗提示词 Skill。训练语义桥，不训练视频 LoRA。推理请安装 [T8 的 MiniMax H3 节点](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)。

1. 从[最新 Release](https://github.com/T8mars/MinimaxH3_Semantic_Bridge_Trainer-T8/releases/latest)下载全部 3 个 `part` 文件、`RELEASE_PARTS.json` 和 `Join-Release.ps1`，放在同一目录。
2. 运行 `powershell -ExecutionPolicy Bypass -File .\Join-Release.ps1`，解压得到的 ZIP，双击 `WushuBridge-Trainer.exe`。便携包自带 Python；训练需自备与 H3 推理一致的文本编码器。
3. 用自己的 `bad/good` 训练对和独立输出目录训练。详见[训练说明](docs/可复用训练器.md)；发行包中的 `plugin/ComfyUI-H3-WushuBridge/` 是旧接线实现，T8 节点的安装和模型目录以[新版推理说明](docs/便携整合包.md#将-semantic-bridge-接入现有-comfyui)为准。

**配套模型：**[T8 Comic Combat Semantic Bridge](https://huggingface.co/t8star/semantic_bridge_T8-comic-combat)。放入 `ComfyUI/models/semantic_bridge/t8_compat/`，使用最新版 T8 节点的 Semantic Bridge 配置／应用节点。实际同提示词对比中，部分复杂连续动作场景有局部改善；建议在自己的工作流中做开／关桥 A/B。

**更多链接：** [B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
