# MiniMax H3 Semantic Bridge Trainer · T8

**Language / 语言:** [简体中文](README.md) · [English](README.en.md)

A reusable **MiniMax H3 Semantic Bridge text-conditioning trainer** with a Windows GUI, CLI training, and an anime combat prompt skill. It trains a semantic bridge, not a video LoRA. For inference, install the [T8 MiniMax H3 nodes](https://github.com/T8mars/comfyui-minimax-h3-audio-T8).

1. Download all three `part` files, `RELEASE_PARTS.json`, and `Join-Release.ps1` from the [latest Release](https://github.com/T8mars/MinimaxH3_Semantic_Bridge_Trainer-T8/releases/latest) into one directory.
2. Run `powershell -ExecutionPolicy Bypass -File .\Join-Release.ps1`, extract the resulting ZIP, and launch `WushuBridge-Trainer.exe`. Python is bundled; supply the same H3 text encoder used for inference.
3. Train on your own `bad/good` pairs in a separate output directory. See the [trainer guide](docs/可复用训练器.md). The bundled `plugin/ComfyUI-H3-WushuBridge/` is a legacy integration; follow the [current inference guide](docs/便携整合包.md#将-semantic-bridge-接入现有-comfyui) for the T8 nodes and model directory.

**Companion model:** [T8 Comic Combat Semantic Bridge](https://huggingface.co/t8star/semantic_bridge_T8-comic-combat). Place it under `ComfyUI/models/semantic_bridge/t8_compat/` and use the latest T8 Semantic Bridge Config/Apply nodes. Same-prompt comparisons showed local improvements in some complex action scenes; compare bridge on/off in your own workflow.

**More links:** [Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
