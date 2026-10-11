# Agent交接：制作用户自己的数据并完成语义桥训练

**版本提示：完整包自带Python/ComfyUI；GitHub NoPython轻量版需先选择已有Python 3.10+及H3 ComfyUI环境。文中 `.\python\python.exe` 命令在轻量版需替换为选定的Python路径。**

## 目标与入口

用户把本包交给你后，按其主题制作bad/good文本偏好数据、检查分组、真实编码和训练Semantic Bridge，最后交付权重与复现说明。先读根目录`AGENTS.md`、`训练集制作.md`和`如何训练.md`。使用本包Python，保留原始文件。

基础数据：`datasets/wushu_pairs_v1_pairs.jsonl`为作者公开676对数值评分数据；`datasets/preference_example.jsonl`的8对仅为格式示例。默认用户自建数据使用无评分偏好模式，不把数值数据、样例或视频反馈标签混在一起。

## A. 只收集必要信息

从用户已有指令推断主题、人物或原创角色、想改善的错误、目标H3工作流、训练预算。缺失的关键问题一次合并询问；能独立做的数据格式、目录与环境检查先做。若用户没有限定角色，可明确采用原创成年动漫角色的假设；不要未经确认替换用户已指定的角色。

先检查本包`models/text_encoders/`和用户提供的编码器位置。必须有与实际部署相同的H3编码器权重；不同编码器或量化版本不是仅凭维度相同就能通用。本包没有大型编码器和视频主模型，不伪造或跳过真实编码。

## B. 制作与审核数据

1. 在`local/<项目名>/`写`TASK.json`，记录用户目标、角色/时期、错误类型、数据规模、编码器文件/量化、来源与待确认项。
2. 利用`skills/minimax-h3-anime-battle-prompts/SKILL.md`及references核实原作事实，设计连续动作、空间转换、遮挡、武器/能力归属与状态持续。若使用命名角色，查一手来源并写`SOURCES.json`；原创编排标明原创，不冒充原作剧情。
3. 写UTF-8 `pairs.jsonl`：每条唯一id、同源group_id、bad/good、pair_preference=good、label_type=authoring_rubric。每对保持同一主题与目标，明确哪些语义错误被纠正，不靠增加无关华丽特效区分优劣。
4. 逐对检查角色、道具、事件一致性、具体纠正点、可执行时长、重复和同源变体。生成行数不是独立场景数。软件至少8对只是门槛，不承诺某个数量能有效。
5. 生成场景分组；所有同源改写/翻译共用group_id，相同文本关联组不跨分区。自动工具需要至少8个可分开的关联场景组：

```powershell
.\python\python.exe examples\preference_starter\make_splits.py local\my_dataset\pairs.jsonl --output local\my_dataset\split_manifest.json
.\python\python.exe tools\trainer.py inspect local\my_dataset\pairs.jsonl --require-supervision preference_only --split-manifest local\my_dataset\split_manifest.json --report local\my_dataset\inspection.json
```

工具无法识别文字不同的同源场景，必须人工正确分组。验证/测试不用于扩大训练或按结果改写；数据已用于开发的事实要记录。将审查摘要写`DATA_REVIEW.md`，区分AI文本审核、真人审核与真正视频观察。

## C. 环境、真实编码与仅语义桥训练

在本包根目录工作，所有示例路径换成用户实际路径；GUI对应字段见`如何训练.md`。GUI首次预填的是8对格式示例，不能冒充用户正式数据；完成自制数据后替换JSONL，普通自编偏好数据保持“自动（推荐）”即可生成分组。group_id仍须由制作数据的Agent正确标注。Python、ComfyUI、缓存、输出和参数已有包内默认路径。

```powershell
.\python\python.exe tools\portable_check.py
.\python\python.exe tools\trainer.py encode local\my_dataset\pairs.jsonl --comfy-root ComfyUI --encoder 'models\text_encoders\用户的H3编码器.safetensors' --output local\my_dataset\pairs_v1.npz
.\python\python.exe tools\trainer.py train local\my_dataset\pairs_v1.npz --stage bridge --split-manifest local\my_dataset\split_manifest.json --config configs\preference_v1.json --output runs\my_run_001
.\python\python.exe tools\trainer.py verify --bridge runs\my_run_001\bridge.safetensors --bridge-only --report local\my_dataset\weight_check.json
.\python\python.exe tools\trainer.py evaluate local\my_dataset\pairs_v1.npz --run runs\my_run_001 --split validation --alpha 0 0.05 0.08 0.12 0.2 --magnitude-mode per_token --output local\my_dataset\validation_per_token.json
```

不要把上面的占位编码器名当成已存在文件。普通train的默认stage是both，所以命令行必须保留`--stage bridge`。配置副本应写到用户自己的local目录并记录哈希；已有NPZ核对训练对和编码器，改变任何一项就选新路径。编码会加载大模型，在资源允许时运行，不停止用户其他任务。

只在同一数据、配置、分组、运行环境和运行目录上用`--resume`续训。基础权重不是checkpoint。默认这是新训练，不擅自覆盖或自动把旧桥当初始化。

## D. 交付与效果验证

- 交付真实`bridge.safetensors`、自动生成的`模型使用说明.md`和`inference_settings.json`、SHA-256、实际配方、数据/分组哈希、编码器身份、训练状态、validation诊断、部署说明；清楚注明文本监督和证据限制。
- 不根据文本指标自动声称最佳alpha。固定契约按权重执行；普通训练仅给出与训练公式一致的试用起点。检查JSON的模型SHA-256与交付权重一致，训练范围不是最佳视频强度。
- 用户部署到T8节点时放到`ComfyUI/models/semantic_bridge/t8_compat/`，使用其Bridge Config/Apply。保留模型自身固定参数契约；不要把旧发布桥的alpha=1.0套到所有新桥。
- 用户授权视频测试时才准备匹配的开/关桥视频，固定提示词、seed、采样与后端；看连续动作、空间、遮挡与对象归属。训练loss、软件测试或样例视频不能代表新训练模型的效果。
- 最终写`local/<项目名>/DELIVERY.md`，列出可点击文件路径、已完成和未完成的项目。不能把未发生的训练/视频检查记成成功。

## 本包附带模型与旧证据

旧动漫Sentence75、作者WUSHU、作者JEV、社区原版桥是不同权重。模型与数据来源见`datasets/SOURCES.json`及各模型报告；JEV不是语义桥。`对照视频/`中的7组14段是已交付的历史局部改善摘选，含不同轮次，不是你新模型的验收结果，也不证明普遍成功率。

## By T8star · 社媒与入口

- [B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [免费画廊](https://www.openzhenzhen.com)
- [在线AI应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI整合包](https://pan.quark.cn/s/264edb7e36bd)
- [Hugging Face](https://huggingface.co/t8star)


## 程序自动更新

启动时自动检查 GitHub，空闲时点“安装更新”即可下载安装并重开。保留 Python、ComfyUI、模型、视频、用户数据、输出和已有配置。详见[自动更新说明](docs/自动更新.md)。轻量 NoPython 包不含 Python 和 ComfyUI，需选择自己现有的运行环境；以下包内 Python 命令仅适用于完整包，轻量包请替换为你选择的 Python 路径。
