# 无评分偏好训练：格式示例

`pairs.format-example.jsonl` 是八对原创动作文本，`split.format-example.json` 是对应的6训练／1验证／1测试场景划分。

**它们只用于格式教学和软件检查，没有经过 H3 视频效果验证。不要把这八条当成正式有效训练集或训练成果。** 每行是一个JSON对象；同一源场景的变体共用group_id。

用户自己的训练流程见 [自己训练从这里开始](../../docs/自己训练从这里开始.md)。先完成自己的文本和审核，再用自己的文件名与路径运行：

```powershell
.\python\python.exe examples\preference_starter\make_splits.py 'D:\MyBridge\pairs.jsonl' --output 'D:\MyBridge\split_manifest.json'
```

`make_splits.py` 不编码、不训练、不自动生成好坏文本。它复用训练器已有的场景／相同文本关联检查，要求至少八个可分开的关联组，将原数值流程的calibration组并入train，为仅语义桥生成约80／10／10的三分区；行数不一定按这个比例。输出文件必须尚不存在。

手动分组时可参考示例JSON的结构，但必须替换成自己的全部group_id。三分区均非空，同一关联组不能跨分区。语义相似但文字不同的同源场景仍需用户主动标注；编码后相同embedding的检查由训练器完成。
