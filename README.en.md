# MiniMax H3 Semantic Bridge Trainer · By T8star

**Language / 语言:** [简体中文](README.md) · [English](README.en.md)

A MiniMax H3 **Semantic Bridge text-conditioning trainer** with Windows EXE, CLI, starter text datasets and an anime combat prompt skill. Some complex actions can improve; it is not a video LoRA and improvement is not guaranteed for every prompt.

1. Download `Windows-NoPython.zip` from the [latest Release](https://github.com/T8mars/MinimaxH3_Semantic_Bridge_Trainer-T8/releases/latest), extract and run `WushuBridge-Trainer.exe`. This lightweight package **does not include Python, ComfyUI, large models or videos**. Existing full-bundle users can update in the launcher.
2. Select an existing **Python 3.10+** environment (PyTorch 2.3+, NumPy, safetensors), an **H3-capable ComfyUI environment**, and the actual H3 text encoder. Full bundles use the bundled runtime. Check the environment before training.
3. Prefer the preference-only bridge mode for custom `bad/good` text pairs. Replace both JSONL and its group split; other defaults are ready. [Training guide](如何训练.md) · [Dataset format](训练集制作.md) · [Agent handoff](AGENT交接.md). Includes 676 public author-scored pairs and 8 format-only examples; the examples are not evidence of video quality.
4. Distribute `bridge.safetensors`, `模型使用说明.md` and `inference_settings.json` together. Follow the model-specific alpha, magnitude and chunk settings. A starting alpha such as 0.12 is not an automatically calibrated optimum.

**Epochs:** Set the bridge epoch limit directly in the launcher (default 150); scored mode also exposes JEV epochs. Early stopping remains enabled. Resume uses the original run limits. Checkpoint writes retry temporary Windows file locks and prevent concurrent writers to the same run.

**Updates:** Startup checks are on by default. Click Install update while idle to verify, install and restart. Python, ComfyUI, models, videos, settings, `local/`, `runs/` and existing config/data files are preserved; current templates are in `defaults/`. [Update and recovery guide](docs/自动更新.md). Pushes to main automatically test, build and publish a lightweight GitHub Release.

**Companion model:** [T8 Comic Combat Semantic Bridge](https://huggingface.co/t8star/semantic_bridge_T8-comic-combat). Install the [T8 MiniMax H3 nodes](https://github.com/T8mars/comfyui-minimax-h3-audio-T8), place weights under `ComfyUI/models/semantic_bridge/t8_compat/`, and use Semantic Bridge Config/Apply nodes. Do not place these in the ordinary LoRA folder. Actual comparisons showed local improvements in some complex action scenes; compare bridge on/off in your own workflow.

**More links:** [B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
