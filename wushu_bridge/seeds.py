"""内置种子提示词：严格按 MiniMax H3 官方两套壳写，并且**自身必须过规则引擎**。

用途
----
1. 插件开箱即用的「正例」来源（用户不提供语料时，靠种子 + 规则降级也能立刻
   造出训练对，先把流水线跑通）；
2. 训练/自测的格式基准；
3. 文档里的示例。

「文生视频」（BASE_SECTIONS）与「多参考图」（REF_SECTIONS）是两套**不同**
字段，绝不能混用 —— 这里分开存放。

写作约束（全部来自 ``wushu_bridge/lint/h3lint.py`` 的实测判据）
--------------------------------------------------------------
* 分镜 **1~3 镜**（4 镜=warn，6 镜以上=error：切太碎）
* ``[Shot 1]`` **不写时间码**，时间码从 ``[Shot 2]`` 开始
* 每个切镜的**开头 170 字**必须交代位置/朝向（在左/在右/面向/间距…）
* 每个切镜要重申锚点（同一张脸 / 同一套服装与武器）
* 正文**不出现** ``【…】``（会被判成残留规则段）
* 正文**不出现**否定式措辞（禁止/不要/不得…）——负面项属于 ComfyUI 的
  negative prompt，不进正向提示词
* 每镜 2~4 句（一拍一句会被判"一镜装不满"）
* 因果连接词 ≥2（随即/因此/于是/趁/被…）
* 开场即交手，不写「对峙」空转
* 不写慢动作/定格/剑气/瞬移/血条 UI

注意：这些种子是为了让流水线可跑、并示范"合格的写法"。真正要效果好，还是要把
用户自己的语料（提示词库、分镜模板、LoRA 打标文件）喂进数据集构建节点。
"""

from __future__ import annotations

from typing import Dict, List

SEED_T2V: List[str] = [
    # ── 对打：雨夜长街，太刀 vs 单刀（3 镜）────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 雨夜长街，两侧灯笼暖光，湿石板反光。角色A是黑发披散的男性，黑色武士劲装，双手持野太刀，画面左前方，间距2格。角色B是灰发束髻的男性，靛蓝汉服武袍，右手持雁翎单刀，画面右后方。此刻角色A的刀尖已经点出，角色B的刀刃横在胸前，两把兵器都挂着水光。
integrated_multimodal_description:
[Shot 1] handheld follow, medium shot，胸口手持跟拍。角色A后脚蹬湿石板，转腰送胯，踏步把间距拉近到1格，普攻出「刀锋点刺」，刀尖走直线刺向角色B胸口，随即接招式「过肩劈」，太刀借着惯性走完整弧线。角色B举刀斜挡，刃对刃撞出火星，因此被压得后退半步，鞋底在湿石上打滑。角色A趁这个空档继续前压，两人仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.100.low angle, tracking，贴地跟拍刀弧。角色B在画面左侧，面向角色A，身位退到2格，侧移半步错开刀路，普攻出「撩刀」，刀自下向上兜起。角色A收刀回架，用刀脊磕住刀身，火星溅起，因此虎口发麻，太刀刀势下沉。角色A随即撤步把间距拉回2格，两人仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.600.over-the-shoulder，过角色A肩看角色B。角色A在左侧，朝向角色B，间距2格，拧身换步绕到角色B左前方，招式「斜劈」让太刀斜下走弧。角色B来不及格挡，硬吃一刀，左肩衣料撕裂，血线渗出，因此踉跄倒退2格，重心压在右腿。角色A跟步送刀，终结技「过肩劈」走完整刀路，角色B举刀硬架却被震脱手，于是沿刀的作用线仰面倒地，湿石水花溅起，不再起身。
overall_soundscape: 踏湿石、刀弧破空、兵刃相交火星、衣料撕裂、闷哼、雨声、粗喘。
non_diegetic_music: None.""",
    # ── 对打：枪 vs 棍，长兵器节奏（3 镜）────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 黄昏演武场，夯土地面扬尘，右侧木栏为实体。角色A是青布短打的女性，持白蜡长枪，画面左侧，间距3格。角色B是赤膊束腰的男性，持齐眉棍，画面右侧。此刻角色A的枪尖已经指向角色B胸口高度，角色B的棍横在身前。
integrated_multimodal_description:
[Shot 1] medium shot, handheld follow。角色A后脚蹬地，腰胯前送，上步把间距拉近到2格，普攻「扎枪」让枪尖走直线刺向胸口高度。角色B横棍封挡，棍身中段磕住枪杆，火星溅起，因此后退半步泄力，前脚掌搓地扬起尘土。角色A趁枪杆被磕开的角度顺势收枪，两人仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.000.low angle，低机位看棍梢。角色B在右侧，面向角色A，身位仍在2格，侧移半步绕开枪线，招式「扫棍」扫向膝部，棍走平弧。角色A提膝避过，用枪尾向下压住棍杆，杆身相碰发出闷响，因此被迫换步。角色A随即撤步把间距拉回3格，两人仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.500.斜侧推进，medium shot。角色A在左侧，朝向角色B，间距3格，转身借腰力反手刺出招式「回马枪」。角色B踏步逼进1格缩短长兵器优势，举棍硬挡，棍梢只擦到枪杆，因此被枪尖掠过肋侧，衣角被带起。角色B顺势前压，终结技「劈棍」从肩后走完整弧线砸向肩头，角色A举枪横架却被砸弯枪杆，于是单膝跪地撑在夯土上，棍梢砸地扬起尘土，角色A仍撑着没有倒地。
overall_soundscape: 夯实脚步、枪杆破空、棍身闷响、尘土落地、衣料摩擦、粗喘。
non_diegetic_music: None.""",
]

SEED_REF2V: List[str] = [
    # ── 多参考图：六段壳，用参考图锁定人物与场景（3 镜）──────────────
    """subject_definitions:
<Subject 1> is the man in <Picture 1>, with loose black hair, a black warrior's jacket, and a long tachi held in both hands. Preserve his face, hairstyle, clothing, and body proportions.
<Subject 2> is the man in <Picture 2>, with tied-back grey hair, an indigo han-style martial robe, and a single yanling saber in his right hand. Preserve his face, hairstyle, clothing, and body proportions.
<Subject 3> is the rainy night street in <Picture 3>, with warm lantern light on both sides and wet reflective stone slabs, and a solid lantern post on the right.
summary:
[reference generation] A 10-second rainy-night duel in which <Subject 1> presses <Subject 2> from two steps to one, and finishes him with an overhead chop that puts him on the ground.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - face, loose black hair, black jacket, and tachi remain unchanged.
<Subject 2> (appears in [Shot 2]): fully_preserved - face, grey tied-back hair, indigo robe, and saber remain unchanged.
<Subject 3> (appears in [Shot 3]): fully_preserved - lantern positions, wet slabs, and the lantern post stay stable.
detailed_description:
[Shot 1] handheld follow, medium shot from chest height. <Subject 1> drives his rear foot into the wet stone, turns his waist, and steps in from two steps to one, then thrusts with the tachi tip along a straight line and follows with an overhead chop that carries the blade through a full arc. <Subject 2> raises his saber into a diagonal guard; edge meets edge, sparks fly, and he is pushed back half a step with his sole sliding on the wet stone. <Subject 1> presses the opening, and the two are still the same two fighters, same faces, same costumes and weapons.
[Shot 2] At 00:02.100.low angle, tracking, following the blade arc close to the ground. <Subject 2> is on the left, facing <Subject 1>, holding two steps of distance, and slides half a step to the side to clear the blade path, then flicks his saber upward. <Subject 1> pulls the tachi back into a guard and jams the blade with his spine; sparks scatter and his grip goes numb, so the tachi sinks. <Subject 1> then steps back to two steps of distance, and the two are still the same two fighters, same faces, same costumes and weapons.
[Shot 3] At 00:05.600.over-the-shoulder from behind <Subject 1>. <Subject 1> is on the left, facing <Subject 2> at two steps, turns his hips and steps around to <Subject 2>'s front-left, and cuts diagonally downward. <Subject 2> cannot guard in time and takes the cut; the cloth on his left shoulder tears and a thin line of blood shows, so he staggers back two steps with his weight on the right leg. <Subject 1> follows through with a finishing overhead chop along the full arc; <Subject 2> braces with his saber but it is knocked from his hands, and he falls onto his back along the line of force, water splashing on the wet stone, and does not get up.
overall_soundscape: feet on wet stone, blade arcs cutting air, steel on steel with sparks, cloth tearing, a muffled grunt, rain, heavy breathing.
non_diegetic_music: None.""",
]

SEED_HORDE: List[str] = [
    # ── 一打多：网格结算后的分镜（3 镜）──────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 夜巷窄街，夯土地面，两侧土墙为实体。角色A是短打劲装持单刀的女性，开场居中，格(10,4)。战场20格宽8格高，可见目标约12个。
integrated_multimodal_description:
[Shot 1] handheld follow, medium shot。角色A后脚蹬地，踏步向前1格到格(9,4)，正前2格有两个目标，普攻「正劈」让刀走垂弧，劈中左侧目标肩头，因此目标侧倒出画，本秒击倒1个。角色A趁刀势回弹拧腰回身，仍是同一人、同一套服装与武器。
[Shot 2] At 00:01.000.low angle，贴地跟刀。右侧贴身一个目标，角色A在格(9,4)朝向格(10,4)，拧腰发力出招式「回身旋转360挥砍」，刀走平圆扫中目标腰侧，因此目标倒退撞墙，本秒击倒1个。角色A随即收刀回到中位，仍是同一人、同一套服装与武器。
[Shot 3] At 00:02.000.wide shot 拉远交代位置。角色A在格(9,4)，格(5,4)附近圈内3个目标，蹬土墙借力跳4格落到格(5,4)，落地屈膝卸力后踏步出招式「横扫」走平弧，连续扫中2个目标，因此两个目标同时倒地。角色A紧接着上步逼近最后1个目标，终结技「过肩劈」走完整刀路，目标沿作用线倒地，不再起身，本场共击倒5个。
overall_soundscape: 踏土、刀风、兵刃入体闷响、倒地、喘息、夜风。
non_diegetic_music: None.""",
]

SEEDS: Dict[str, List[str]] = {
    "t2v": SEED_T2V,
    "ref2v": SEED_REF2V,
    "horde": SEED_HORDE,
}


def all_seeds(mode: str = "t2v") -> List[str]:
    """按模式取种子。``t2v`` 模式会带上 horde 样例（同样是文生视频壳）。"""
    if mode == "ref2v":
        return list(SEED_REF2V)
    return list(SEED_T2V) + list(SEED_HORDE)
