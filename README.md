# MiniMax H3 Semantic Bridge Trainer · By T8star

**语言 / Language：** [简体中文](README.md) · [English](README.en.md)

MiniMax H3 **语义桥文本条件训练器**，包含 Windows EXE、命令行、基础文本数据和动漫战斗提示词 Skill。用于改善部分复杂动作的语义执行；不是视频 LoRA，不能保证每条提示词都改善。

1. 从[最新 Release](https://github.com/T8mars/MinimaxH3_Semantic_Bridge_Trainer-T8/releases/latest)下载 `Windows-NoPython.zip`，解压后双击 `WushuBridge-Trainer.exe`。这是轻量包，**不含 Python、ComfyUI、大模型和视频**。已有完整包用户直接使用启动器更新。
2. 轻量包选择现有 **Python 3.10+**（PyTorch 2.3+、NumPy、safetensors）、可运行 H3 编码的 **ComfyUI 环境**和实际 H3 文本编码器。完整包自动使用内置环境。依赖检查通过后再训练，不自动安装大模型或修改其他项目。
3. 自己的数据优先选“无评分偏好对：仅语义桥”，更换 JSONL 和配套分组，其余保持默认。[怎么训练](如何训练.md) · [数据格式与示例](训练集制作.md) · [交给 Agent 制作并训练](AGENT交接.md)。内置676对作者公开评分文本和8对格式示例；8对仅供学习格式。
4. 输出 `bridge.safetensors`、`模型使用说明.md`、`inference_settings.json`。按该模型的说明设置 alpha、幅度模式和 chunk；普通配方的0.12是试用起点，不是自动验证的最佳值。

**自动更新：** 默认启动时检查 Release，空闲时点“安装更新”，校验后安装并重启。保留 Python、ComfyUI、模型、视频、`local/`、`runs/`、路径设置及已有配置/数据；新版模板在 `defaults/`。[更新与恢复说明](docs/自动更新.md)。推送到 main 会通过 GitHub Actions 自动测试、编译、发布新的轻量 Release。

**配套模型：** [T8 Comic Combat Semantic Bridge](https://huggingface.co/t8star/semantic_bridge_T8-comic-combat)。推理安装 [T8 MiniMax H3 节点](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)，模型放 `ComfyUI/models/semantic_bridge/t8_compat/`。使用 Semantic Bridge Config／Apply 节点接入文本条件，不放普通 `models/loras/`。实际测试部分复杂动作场景有局部改善，建议按自己的工作流做开／关桥 A/B。

**更多链接：** [B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
