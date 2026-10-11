# 本整合包的Agent工作规则

先读AGENT交接.md，再按用户目标工作。默认任务是制作文本偏好数据并训练Semantic Bridge，不训练H3视频LoRA，不修改视频主模型或编码器。

- 优先使用本包python/python.exe；NoPython轻量版使用用户指定的现有Python与H3 ComfyUI环境，命令行以本包根目录为工作目录。所有用户数据、缓存与结果放在本包新建的local/和runs/，原始模型、基础数据与样例不覆盖。
- 用户已要求“制作数据并训练”时，在必要输入齐备后完成检查、编码、仅语义桥训练与交付，不反复索取同一授权；若用户说“先不训练”，只准备数据。
- 普通数据使用authoring_rubric、pair_preference=good和正确group_id，不伪造数值评分、真实视频标签或原作事实。命名角色先核实一手资料，并记录来源。
- 先检查文本和分组，新的数据或编码器使用新的NPZ和运行目录。默认train必须显式--stage bridge，不启动JEV或视频LoRA流程。
- local/、runs/、个人设置、令牌和用户私有数据不上传，不进之后的整合包。若工作区已有roadmap.md，保持私人本地，不stage/commit/upload/package。
- 软件测试、格式检查、文本指标和训练完成均不等于实际视频质量；报告真实范围。不要以特效更亮代替动作关系改善。
- 不擅自停止其他任务、卸载其他项目模型或更新另一项目源码。安装部署与视频评估按用户范围进行。

- 轻量NoPython版使用用户选择的Python与ComfyUI路径，不假设已包含python/或ComfyUI/。更新程序不得打包Python、模型、local/、runs/或roadmap.md，defaults/仅放公开模板。
