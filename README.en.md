# MiniMax H3 Semantic Bridge Trainer · T8

**Language / 语言:** [简体中文](README.md) · [English](README.en.md)

A reusable **MiniMax H3 Semantic Bridge text-conditioning trainer** with a Windows GUI, CLI training, a ComfyUI integration plugin, and an anime combat prompt skill. It trains a semantic bridge, not a video LoRA.

1. Download all three `part` files, `RELEASE_PARTS.json`, and `Join-Release.ps1` from the [latest Release](https://github.com/T8mars/MinimaxH3_Semantic_Bridge_Trainer-T8/releases/latest) into one directory.
2. Run `powershell -ExecutionPolicy Bypass -File .\Join-Release.ps1`, extract the resulting ZIP, and launch `WushuBridge-Trainer.exe`. Python is bundled; supply the same H3 text encoder used for inference.
3. Train on your own `bad/good` pairs in a separate output directory. See the [trainer guide](docs/可复用训练器.md); the ComfyUI plugin source is in `plugin/ComfyUI-H3-WushuBridge/`.

**Companion model:** [T8 Comic Combat Semantic Bridge](https://huggingface.co/t8star/semantic_bridge_T8-comic-combat). Insert this weight between H3 text encoding and video sampling. Same-prompt comparisons showed local improvements in some complex action scenes; compare bridge on/off in your own workflow.

**More links:** [Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
