#!/usr/bin/env python3
"""R4 真实训练风格的 128 条四分类诊断样例。

样例措辞和关系类型取自用户提供的两份真实 summary 文本，覆盖人物/物体的
相对位置、朝向、远近、高度、出现/消失、替换、门内外、背景变化、群体关系
以及因视角遮挡导致的不确定性。

32 个基础场景各派生 A/B/C/D 一条，共 128 条。排列经过固定轮换，使每个
连续 32 条 batch 都恰好包含 8 条 A、8 条 B、8 条 C、8 条 D。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


EXPECTED_CLASSES = ("A", "B", "C", "D")
DIAGNOSTIC_BATCH_SIZE = 32


@dataclass(frozen=True)
class DiagnosticCase:
    """一条带人工预期类别的 summary 对。"""

    pred_summary: str
    gt_summary: str
    expected_class: str
    description: str


# 每项依次为：场景名、reference、A candidate、B candidate、C candidate、
# D candidate。A/B/C/D 的定义与 reward_model.py 中的 prompt 完全一致。
_SCENARIOS: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    (
        "进门并转向",
        "The man moved from outside the doorway to inside the room, and turned from facing away from the door to facing it.",
        "The man entered the room through the doorway and turned around so that he now faces the door instead of away from it.",
        "The man went from outside the doorway to inside the room.",
        "The man entered the room, but he continued facing away from the door.",
        "The man left the room through the doorway and turned his back toward the door.",
    ),
    (
        "沙发距离与房间背景",
        "The woman became closer to the sofa, and the background changed from a living room to a study.",
        "The woman moved nearer the sofa while the setting switched from the living room to a study.",
        "The woman moved nearer to the sofa.",
        "The woman moved closer to the sofa, and the living room changed into a cafe.",
        "The woman moved farther from the sofa, while the scene remained the same living room.",
    ),
    (
        "人物新增与儿童消失",
        "An extra woman appeared on the left side of the man, and the child between the couple disappeared.",
        "A new woman showed up to the man's left, while the child who had stood between the two adults was gone.",
        "A new woman appeared to the left of the man.",
        "A woman appeared on the man's left, and a second child appeared between the couple.",
        "The woman on the man's right disappeared, and the child remained between the couple.",
    ),
    (
        "三人队形与汽车距离",
        "The three people changed from a triangular arrangement to standing side-by-side, and all three moved farther away from the car.",
        "All three people moved away from the car and rearranged from a triangle into a side-by-side line.",
        "The three people now stand side-by-side instead of in a triangle.",
        "The group changed from a triangle to a side-by-side line, but all three moved closer to the car.",
        "The three people formed a triangle after standing side-by-side and moved closer to the car.",
    ),
    (
        "两名女性方位与朝向",
        "The two women changed from facing each other to standing side-by-side, and the woman in white moved from the left of the woman in pink to her right.",
        "The women no longer face one another and instead stand next to each other; the white-clothed woman also switched from the pink-clothed woman's left side to her right.",
        "The two women changed from face-to-face to side-by-side.",
        "The women became side-by-side, but the woman in white stayed to the left of the woman in pink.",
        "The women went from side-by-side to face-to-face, and the woman in white moved from right to left of the woman in pink.",
    ),
    (
        "床的距离与人物高度",
        "The man and the woman became farther from the bed, and the woman's relative height became lower as she changed from standing to sitting.",
        "Both people moved away from the bed, and the woman went from standing to sitting, making her lower.",
        "The man and woman moved farther away from the bed.",
        "They moved farther from the bed, while the woman rose from sitting to standing and became taller.",
        "They moved closer to the bed, and the woman stood up from her seat and became higher.",
    ),
    (
        "人物替换与柱子方位",
        "The woman in the trench coat was replaced by a man in black, who stood to the right of the red pillar.",
        "A black-clothed man took the place of the trench-coat woman and was positioned on the red pillar's right side.",
        "The trench-coat woman was replaced by a man dressed in black.",
        "A man in black replaced the trench-coat woman, but he stood to the left of the red pillar.",
        "The woman in the trench coat remained and stood on the left side of the red pillar.",
    ),
    (
        "构图一致仅视角变化",
        "The character positions and the scene are consistent; only the shot scale and camera perspective are different.",
        "The people and setting match spatially, with the apparent difference caused solely by framing and viewpoint.",
        "The arrangement of the characters is spatially consistent.",
        "The character layout is consistent, but the background is a completely different location rather than a perspective change.",
        "The characters occupy different positions and the scene itself has changed substantially.",
    ),
    (
        "动物增减与距离",
        "A pink animal appeared to the left of the blue animal, the blue animal moved closer to the green animal, and the red animal disappeared.",
        "A new pink animal showed up on the blue animal's left; the blue and green animals became nearer, while the red animal vanished.",
        "A pink animal appeared on the left side of the blue animal.",
        "The pink animal appeared left of the blue one, but the red animal remained and moved closer to them.",
        "The pink animal disappeared from the right of the blue animal, the blue moved away from the green, and the red animal stayed.",
    ),
    (
        "门外位置与人物次序",
        "The woman in green and the two maids moved from inside the door to outside it, with the woman in green standing in front of both maids.",
        "The green-clothed woman and both maids went out through the door, and she ended up ahead of the two maids.",
        "The woman in green and both maids moved from inside the doorway to outside.",
        "All three moved outside the door, but the green-clothed woman stood behind the maids.",
        "All three entered through the door, with the woman in green following behind the two maids.",
    ),
    (
        "男女换位并远离",
        "The man and woman swapped their left-right positions, and the distance between them became farther.",
        "The two people exchanged sides and moved farther apart from each other.",
        "The man and woman switched which side each person stood on.",
        "They swapped left-right positions, but the distance between them became closer.",
        "They stayed on their original sides and moved closer together.",
    ),
    (
        "女性转身并变高",
        "The woman changed from facing the man to having her back to him, and her relative height became taller as she stood up.",
        "The woman stood up and became higher while turning from face-to-face with the man to facing away from him.",
        "The woman turned from facing the man to having her back toward him.",
        "She turned her back to the man, but became lower by sitting down.",
        "She remained facing the man and sat down, making her lower.",
    ),
    (
        "室内外背景与人物缺失",
        "The background changed from an indoor office to an outdoor road, and the woman in yellow disappeared.",
        "The setting moved outdoors onto a road instead of inside the office, and the yellow-clothed woman was no longer present.",
        "The office background changed to an outdoor road.",
        "The scene changed from the office to an outdoor road, but an extra woman in yellow appeared.",
        "The setting stayed inside the office, and the yellow-clothed woman remained visible.",
    ),
    (
        "桌旁坐立与远近",
        "The man changed from sitting beside the table to standing farther away from it, making his relative height higher.",
        "The man stood up from his seat by the table, became taller, and moved farther from the table.",
        "The seated man stood up and therefore became relatively higher.",
        "The man stood up and became taller, but moved closer to the table.",
        "The man sat down closer to the table and became lower.",
    ),
    (
        "祭坛距离与新增人物",
        "The old man moved closer to the memorial altar, and an extra man in brown appeared behind him on the right.",
        "The elderly man approached the altar, while a new brown-clothed man appeared at his rear right.",
        "The old man became nearer to the memorial altar.",
        "The old man moved closer to the altar, but the man in brown appeared in front of him on the left.",
        "The old man moved away from the altar, and the brown-clothed man disappeared from his front left.",
    ),
    (
        "遮挡导致存在不确定",
        "The visible character positions are consistent, but Image A does not show whether a brown bear stands opposite the woman, so the bear's presence is uncertain.",
        "The people that can be seen are positioned consistently; because the relevant area is hidden in Image A, it is impossible to tell if the brown bear is opposite the woman.",
        "The positions of all visible characters are consistent.",
        "The visible people are positioned consistently, and the brown bear is definitely absent from opposite the woman.",
        "The visible characters are arranged differently, and Image A clearly proves that the bear is absent.",
    ),
    (
        "人群消失与局部背景",
        "The crowd behind the woman disappeared, and the wall on the left changed into a doorway with curtains.",
        "The people behind the woman were gone, while the left-side wall was replaced by a curtained doorway.",
        "The crowd that had been behind the woman disappeared.",
        "The rear crowd disappeared, but the left wall remained unchanged without a doorway.",
        "More people appeared behind the woman, and the curtained doorway changed back into a plain wall.",
    ),
    (
        "面对面变前后站位",
        "The man and woman changed from facing each other to standing one behind the other, with the man behind the woman.",
        "The pair no longer face one another; they are aligned front-to-back with the woman ahead of the man.",
        "The man and woman changed from face-to-face to a front-and-back arrangement.",
        "They became aligned front-to-back, but the woman stood behind the man.",
        "They changed from a front-and-back arrangement to facing each other, with the man in front.",
    ),
    (
        "多目标相对距离",
        "The woman in purple and the man in yellow both moved farther from the daybed, while becoming closer to each other.",
        "Both people increased their distance from the daybed, even as the purple-clothed woman and yellow-clothed man moved nearer one another.",
        "The woman in purple and the man in yellow moved farther away from the daybed.",
        "They moved farther from the daybed, and also moved farther apart from each other.",
        "They moved closer to the daybed while increasing the distance between themselves.",
    ),
    (
        "两人换位与第三人转向",
        "The woman in blue and the man in green swapped positions, and the woman in pink changed from facing forward to facing backward.",
        "The blue-clothed woman exchanged places with the green-clothed man, while the pink-clothed woman turned from front-facing to back-facing.",
        "The woman in blue and the man in green exchanged their positions.",
        "The two swapped positions, but the woman in pink continued facing forward.",
        "The blue-clothed woman and green-clothed man stayed in place, while the woman in pink turned from backward to forward.",
    ),
    (
        "小狗朝向与门的距离",
        "The puppy changed from facing the door to facing away from it, and moved farther from the doorway.",
        "The dog turned its back on the door and increased its distance from the doorway.",
        "The puppy turned from facing the door to facing away from it.",
        "The puppy faced away from the door but moved closer to the doorway.",
        "The puppy turned toward the door and approached the doorway.",
    ),
    (
        "物体新增与老人消失",
        "An opened umbrella and a trident appeared beside the woman, and the old man opposite her disappeared.",
        "A new open umbrella and trident showed up next to the woman, while the elderly man facing her vanished.",
        "An open umbrella and a trident appeared beside the woman.",
        "The umbrella and trident appeared, but the old man remained opposite the woman.",
        "The umbrella and trident disappeared, and an old man newly appeared behind the woman.",
    ),
    (
        "车辆移动与局部背景替换",
        "The two vehicles moved toward the camera, and the yellow car in the background was replaced by a tent.",
        "Both vehicles came closer to the viewpoint, while a tent took the place of the background's yellow car.",
        "The two vehicles moved toward the camera.",
        "Both vehicles approached the camera, but the tent was replaced by a yellow car.",
        "The vehicles moved away from the camera, and the yellow car remained in the background.",
    ),
    (
        "人物替换与场景切换",
        "The white-haired man was replaced by a bear, and the background changed from a wheat field to a bamboo forest.",
        "A bear took the white-haired man's place as the setting switched from a wheat field to a bamboo forest.",
        "The white-haired man was replaced by a bear.",
        "A bear replaced the white-haired man, but the background changed from a wheat field to a city street.",
        "The white-haired man remained, and the bamboo forest changed into a wheat field.",
    ),
    (
        "群体成员增减",
        "Four masked people in black appeared, while the spellcaster and the person in red disappeared.",
        "Four new black-clothed masked figures arrived, and both the spellcaster and red-clothed person vanished.",
        "Four masked people dressed in black appeared.",
        "The four masked people appeared, but the spellcaster and person in red both remained.",
        "The masked group disappeared, while the spellcaster and person in red newly appeared.",
    ),
    (
        "家具替换与人物位置",
        "The computer desk in front of the woman became a bed, and the woman moved from beside the door to beside the window.",
        "A bed replaced the computer desk ahead of the woman, who relocated from the doorway to the window.",
        "The computer desk in front of the woman was replaced by a bed.",
        "The desk became a bed, but the woman stayed beside the door rather than moving to the window.",
        "The bed changed into a computer desk, and the woman moved from the window back to the door.",
    ),
    (
        "三项关系组合",
        "The man and woman moved closer and changed from side-by-side to face-to-face, while the child between them disappeared.",
        "The pair became nearer and turned from standing shoulder-to-shoulder to facing each other; the child in the middle was gone.",
        "The man and woman moved closer and became face-to-face instead of side-by-side.",
        "The pair moved closer and became face-to-face, but a second child appeared between them.",
        "The pair moved farther apart and changed from face-to-face to side-by-side, while the child remained between them.",
    ),
    (
        "人物骑马与场景变化",
        "The background changed from indoors to outdoors, and the man and woman changed from standing to riding horses.",
        "The scene moved outside from an interior, with both people now mounted on horses instead of standing.",
        "The indoor background changed to an outdoor setting.",
        "The setting became outdoors, but the man and woman remained standing beside the horses.",
        "The scene moved from outdoors to indoors, and both people dismounted to stand on the ground.",
    ),
    (
        "门的位置与女性朝向",
        "The woman moved from inside the room to outside the door and changed from facing the door to having her back to it.",
        "The woman exited through the doorway and turned around so her back, rather than her face, was toward the door.",
        "The woman moved from inside the room to outside the doorway.",
        "The woman moved outside, but she continued facing the door.",
        "The woman entered the room and turned to face the door.",
    ),
    (
        "一致场景与取景遗漏",
        "The standing positions and scene are consistent; Image A is simply a close-up that does not capture the woman farther behind.",
        "The spatial arrangement and setting match, but the woman in the rear falls outside Image A's tighter crop.",
        "The visible character positions and the setting are consistent.",
        "The visible arrangement is consistent, but the rear woman was deleted rather than merely outside the close-up.",
        "The character positions and setting conflict, and both images clearly include the rear woman.",
    ),
    (
        "多人站位与新增角色",
        "The man in grey moved to the left of the man with glasses, and an extra woman in white appeared behind them.",
        "The grey-clothed man shifted to the glasses-wearing man's left, with a new white-clothed woman showing up behind the pair.",
        "The man in grey moved to the left side of the man with glasses.",
        "The grey-clothed man moved left of the man with glasses, but the woman in white appeared in front of them.",
        "The man in grey moved to the right of the man with glasses, and the white-clothed woman disappeared from behind them.",
    ),
    (
        "人物姿态与走廊背景",
        "The man in black changed both position and orientation, and the background changed from beside the door to an outdoor hallway.",
        "The black-clothed man relocated and turned to a different facing direction, while the doorway setting became an exterior corridor.",
        "The man in black changed his position and facing direction.",
        "The man in black changed position and orientation, but the background remained beside the same indoor door.",
        "The man kept the same position and orientation, and the outdoor hallway changed back to the indoor doorway.",
    ),
)


# 避免每个场景机械地按 A/B/C/D 排列，同时保证每四个场景、乃至每个
# 32 条 batch 都类别均衡。
_CLASS_ORDERS = (
    ("A", "B", "C", "D"),
    ("B", "D", "A", "C"),
    ("C", "A", "D", "B"),
    ("D", "C", "B", "A"),
)


def build_realistic_diagnostic_cases() -> List[DiagnosticCase]:
    """构造并严格校验 128 条、32 条一批的均衡诊断集。"""
    if len(_SCENARIOS) != 32:
        raise RuntimeError(
            f"R4 diagnostic scenarios must contain 32 items, got {len(_SCENARIOS)}"
        )

    cases: List[DiagnosticCase] = []
    for scenario_index, scenario in enumerate(_SCENARIOS):
        if len(scenario) != 6:
            raise RuntimeError(
                f"R4 diagnostic scenario {scenario_index + 1} must contain "
                f"a description, reference, and four candidates; got {len(scenario)} fields"
            )
        description, gt_summary, *candidate_summaries = scenario
        candidates = dict(zip(EXPECTED_CLASSES, candidate_summaries))
        for expected_class in _CLASS_ORDERS[scenario_index % len(_CLASS_ORDERS)]:
            pred_summary = candidates[expected_class]
            if " ".join(pred_summary.split()).casefold() == " ".join(
                gt_summary.split()
            ).casefold():
                raise RuntimeError(
                    f"R4 diagnostic {description!r}/{expected_class} uses the "
                    "exact-match fast path"
                )
            cases.append(
                DiagnosticCase(
                    pred_summary=pred_summary,
                    gt_summary=gt_summary,
                    expected_class=expected_class,
                    description=description,
                )
            )

    expected_total = 128
    if len(cases) != expected_total:
        raise RuntimeError(
            f"R4 diagnostic cases must contain {expected_total} items, got {len(cases)}"
        )

    for batch_start in range(0, len(cases), DIAGNOSTIC_BATCH_SIZE):
        batch = cases[batch_start : batch_start + DIAGNOSTIC_BATCH_SIZE]
        class_counts = {
            letter: sum(case.expected_class == letter for case in batch)
            for letter in EXPECTED_CLASSES
        }
        if len(batch) != DIAGNOSTIC_BATCH_SIZE or any(
            count != DIAGNOSTIC_BATCH_SIZE // len(EXPECTED_CLASSES)
            for count in class_counts.values()
        ):
            raise RuntimeError(
                f"R4 diagnostic batch {batch_start // DIAGNOSTIC_BATCH_SIZE + 1} "
                f"is not balanced: size={len(batch)}, classes={class_counts}"
            )

    return cases
