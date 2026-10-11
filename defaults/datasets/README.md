# 基础数据

| 文件 | 内容 | 训练方式 |
|---|---|---|
| wushu_pairs_v1_pairs.jsonl | 作者公开676对武术文本，包含bad/good及数值分数 | 数值评分数据：语义桥+JEV |
| preference_example.jsonl | 8对原创文本格式示例，没有视频效果验证 | 无评分偏好：仅语义桥；主要用于格式教学 |
| preference_example_split.json | 与8条示例匹配的6训练/1验证/1测试分组 | 不要直接套用到用户的新数据 |
| SOURCES.json | 来源、文件哈希与性质 | 用于核对与溯源 |

作者清单同时在历史模型布局models/wushu_bridge/datasets/中保留副本，保证原GUI数值模式默认路径可用。两份676对逐字节一致。

用户的新数据请另建local/<项目名>/，按训练集制作.md整理。所有基础文件保持原样。
