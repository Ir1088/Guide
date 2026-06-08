from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = REPO_ROOT / "data" / "knowledge_base" / "chunks" / "chunks.jsonl"
MANIFEST_FILE = REPO_ROOT / "data" / "knowledge_base" / "chunks" / "manifest.json"
QA_FILE = REPO_ROOT / "data" / "knowledge_base" / "qa_test_set.json"


SPOT_AUDIENCE = {
    "灵山大照壁": ["all", "photo", "culture"],
    "五明桥": ["all", "photo", "culture"],
    "佛足坛": ["culture", "blessing"],
    "五智门": ["culture", "blessing"],
    "菩提大道": ["all", "photo"],
    "九龙灌浴": ["all", "family", "performance", "photo"],
    "降魔浮雕": ["culture", "family"],
    "阿育王柱": ["culture", "photo"],
    "百子戏弥勒": ["family", "photo"],
    "祥符禅寺": ["culture", "history", "blessing"],
    "灵山大佛": ["all", "culture", "photo", "blessing"],
    "佛教文化博览馆": ["culture", "family", "education"],
    "灵山梵宫": ["culture", "performance", "photo"],
    "五印坛城": ["culture", "blessing", "photo"],
    "曼飞龙塔": ["culture", "photo"],
    "无尽意斋": ["culture", "dining"],
    "拈花广场": ["all", "route", "photo"],
    "梵天花海": ["photo", "leisure"],
    "香月花街": ["leisure", "shopping", "dining"],
    "拈花堂": ["culture", "quiet", "photo"],
    "五灯湖": ["leisure", "night", "photo"],
    "鹿鸣谷": ["family", "nature", "photo"],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def infer_topic(chunk: dict[str, Any]) -> str:
    text = "\n".join(
        [
            chunk.get("title", ""),
            chunk.get("section", ""),
            " ".join(chunk.get("keywords", [])),
            chunk.get("content", ""),
        ]
    )
    if chunk.get("document_id") == "tourism_behavior_summary":
        return "behavior"
    if "路线" in text:
        return "route"
    if "演出" in text or "开放" in text or "时间" in text:
        return "schedule"
    if "费用" in text or "免费" in text or "门票" in text or "消费" in text:
        return "price"
    if "亲子" in text:
        return "family"
    if "拍照" in text or "打卡" in text:
        return "photo"
    if "历史" in text or "玄奘" in text or "赵朴初" in text:
        return "history"
    if "文化" in text or "佛教" in text:
        return "culture"
    return "general"


def enrich_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})
    spot_name = metadata.get("spot_name") or (chunk.get("title") if chunk.get("title") in SPOT_AUDIENCE else "")
    if spot_name:
        metadata["spot_name"] = spot_name
    metadata.setdefault("topic", infer_topic(chunk))
    metadata.setdefault("audience", SPOT_AUDIENCE.get(spot_name, ["all"]))
    metadata.setdefault(
        "data_scope",
        "partial" if chunk.get("document_id") == "tourism_behavior_summary" else "scenic_knowledge_base",
    )
    chunk["metadata"] = metadata
    return chunk


def fact_chunk(
    chunk_id: str,
    title: str,
    topic: str,
    content: str,
    *,
    spot_name: str = "",
    audience: list[str] | None = None,
    data_scope: str = "scenic_knowledge_base",
    keywords: list[str] | None = None,
    source: str = "curated_facts",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "document_type": "curated_fact",
        "topic": topic,
        "audience": audience or ["all"],
        "data_scope": data_scope,
    }
    if spot_name:
        metadata["spot_name"] = spot_name
        metadata["scenic_area"] = "灵山胜境" if not spot_name.startswith("拈花") else "拈花湾禅意小镇"
    return {
        "chunk_id": chunk_id,
        "document_id": "ling_shan_curated_facts",
        "title": title,
        "section": topic,
        "source_file": "data/knowledge_base/curated_facts/rag_baseline_facts.md",
        "source_type": "curated_fact",
        "keywords": keywords or [title, topic],
        "metadata": metadata,
        "citation": f"data/knowledge_base/curated_facts/rag_baseline_facts.md > {title} > {chunk_id}",
        "content": content,
    }


FACTS = [
    fact_chunk(
        "fact-location-001",
        "灵山胜境位置",
        "location",
        "灵山胜境坐落于江苏省无锡市太湖西北部的马山镇，依托小灵山、太湖山水和佛教文化资源建设。",
        keywords=["灵山胜境", "位置", "无锡", "马山镇", "太湖"],
    ),
    fact_chunk(
        "fact-origin-001",
        "小灵山命名由来",
        "history",
        "玄奘法师见马山山形酷似印度灵鹫山，认为此地与佛法有缘，赐名为小灵山。",
        keywords=["小灵山", "玄奘", "灵鹫山", "命名"],
    ),
    fact_chunk(
        "fact-history-001",
        "祥符禅寺历史沿革",
        "history",
        "祥符禅寺源于小灵山庵，唐贞观年间已有佛教活动，北宋大中祥符年间获赐额祥符禅寺，历经兴废，是灵山胜境的千年文化根基。",
        spot_name="祥符禅寺",
        audience=["history", "culture"],
        keywords=["祥符禅寺", "小灵山庵", "北宋", "大中祥符", "历史"],
    ),
    fact_chunk(
        "fact-buddha-001",
        "灵山大佛落成开光",
        "history",
        "灵山大佛于1997年11月15日落成开光，是灵山胜境的核心地标。",
        spot_name="灵山大佛",
        audience=["history", "culture"],
        keywords=["灵山大佛", "1997年11月15日", "落成", "开光"],
    ),
    fact_chunk(
        "fact-zhaopuchu-001",
        "赵朴初与灵山胜境",
        "history",
        "赵朴初提出五方五佛理念，推动灵山大佛文化格局形成，并题写灵山胜境大照壁。",
        spot_name="灵山大照壁",
        audience=["history", "culture"],
        keywords=["赵朴初", "五方五佛", "灵山大照壁", "题写"],
    ),
    fact_chunk(
        "fact-dazhaobi-001",
        "灵山大照壁游玩价值",
        "photo",
        "灵山大照壁是景区入口标志性门户，适合入园第一站打卡合影，也适合解读赵朴初题字和《小灵山》诗刻文化。",
        spot_name="灵山大照壁",
        audience=["photo", "culture"],
        keywords=["灵山大照壁", "大照壁", "打卡", "合影", "诗刻"],
    ),
    fact_chunk(
        "fact-wumingqiao-001",
        "五明桥文化含义",
        "culture",
        "五明桥象征佛教五种核心智慧：声明、因明、内明、医方明、工巧明，寓意过桥开启智慧、走向觉悟。",
        spot_name="五明桥",
        audience=["culture"],
        keywords=["五明桥", "五明", "智慧", "声明", "因明", "内明", "医方明", "工巧明"],
    ),
    fact_chunk(
        "fact-jiulong-001",
        "九龙灌浴核心看点",
        "performance",
        "九龙灌浴以释迦牟尼诞生典故为核心，通过莲花绽放、太子佛升起和九龙喷水演绎“花开见佛，九龙沐浴”的动态景观。",
        spot_name="九龙灌浴",
        audience=["all", "family", "photo"],
        keywords=["九龙灌浴", "佛祖诞生", "释迦牟尼", "动态景观", "花开见佛"],
    ),
    fact_chunk(
        "fact-ayuwang-001",
        "阿育王柱文化意义",
        "culture",
        "阿育王柱复刻古印度阿育王石柱造型，四狮柱头象征佛法向四方传播，体现佛教传播、和平包容和护法精神。",
        spot_name="阿育王柱",
        audience=["culture", "photo"],
        keywords=["阿育王柱", "佛教传播", "四狮", "护法精神"],
    ),
    fact_chunk(
        "fact-xiangfu-001",
        "祥符禅寺游览价值",
        "culture",
        "祥符禅寺承载千年佛教传承，是灵山胜境核心文化节点，适合礼佛祈福、聆听禅钟、了解唐宋以来的禅宗历史。",
        spot_name="祥符禅寺",
        audience=["culture", "history", "blessing"],
        keywords=["祥符禅寺", "千年佛教", "礼佛", "禅钟", "文化节点"],
    ),
    fact_chunk(
        "fact-buddha-002",
        "灵山大佛游玩亮点",
        "experience",
        "灵山大佛的游玩亮点包括瞻仰88米露天大佛、登临佛脚平台观景、体验抱佛脚祈福，并理解五方五佛文化格局。",
        spot_name="灵山大佛",
        audience=["all", "culture", "photo", "blessing"],
        keywords=["灵山大佛", "瞻仰", "抱佛脚", "观景", "五方五佛"],
    ),
    fact_chunk(
        "fact-museum-001",
        "佛教文化博览馆适合人群",
        "audience",
        "佛教文化博览馆适合想深入了解佛教历史、佛教艺术、四大名山和世界佛教传播脉络的游客，也适合亲子科普和研学参观。",
        spot_name="佛教文化博览馆",
        audience=["culture", "family", "education"],
        keywords=["佛教文化博览馆", "适合人群", "佛教历史", "亲子", "研学"],
    ),
    fact_chunk(
        "fact-fangong-001",
        "灵山梵宫特色",
        "culture",
        "灵山梵宫融合佛教艺术、传统工艺和现代科技，汇集东阳木雕、琉璃、油画、景泰蓝等艺术，并以星空穹顶和《灵山吉祥颂》增强沉浸体验。",
        spot_name="灵山梵宫",
        audience=["culture", "performance", "photo"],
        keywords=["灵山梵宫", "佛教艺术", "传统工艺", "现代科技", "灵山吉祥颂"],
    ),
    fact_chunk(
        "fact-wuyin-001",
        "五印坛城文化",
        "culture",
        "五印坛城体现藏传佛教文化与坛城艺术，“五印”对应五方五佛手印，坛城象征宇宙和谐、圆满与神圣。",
        spot_name="五印坛城",
        audience=["culture", "blessing"],
        keywords=["五印坛城", "藏传佛教", "坛城", "五方五佛"],
    ),
    fact_chunk(
        "fact-manfeilong-001",
        "曼飞龙塔特色",
        "culture",
        "曼飞龙塔复刻云南西双版纳曼飞龙白塔，展现南传佛教建筑风格和多元佛教文化，是拍摄九塔组合与了解南传佛教的特色点位。",
        spot_name="曼飞龙塔",
        audience=["culture", "photo"],
        keywords=["曼飞龙塔", "南传佛教", "白塔", "九塔", "多元佛教"],
    ),
    fact_chunk(
        "fact-family-route-001",
        "亲子游客推荐点位",
        "route",
        "亲子游客可优先关注九龙灌浴、百子戏弥勒、佛教文化博览馆、灵山大佛和灵山梵宫；这些点位兼具动态表演、互动拍照、文化科普和轻松游览。",
        audience=["family"],
        keywords=["亲子", "九龙灌浴", "百子戏弥勒", "佛教文化博览馆", "灵山大佛", "灵山梵宫"],
    ),
    fact_chunk(
        "fact-photo-route-001",
        "灵山胜境拍照打卡路线",
        "route",
        "拍照打卡可按灵山大照壁、五明桥、九龙灌浴、灵山大佛、灵山梵宫安排，依次覆盖入口门户、水景桥面、动态演艺、核心地标和建筑艺术场景。",
        audience=["photo"],
        keywords=["拍照", "打卡", "灵山大照壁", "五明桥", "九龙灌浴", "灵山大佛", "灵山梵宫"],
    ),
    fact_chunk(
        "fact-culture-route-001",
        "历史文化深度路线",
        "route",
        "历史文化深度游可按灵山大照壁、祥符禅寺、灵山大佛、佛教文化博览馆、灵山梵宫、五印坛城安排，重点理解玄奘、小灵山、五方五佛和三大语系佛教建筑。",
        audience=["culture", "history"],
        keywords=["历史路线", "文化路线", "祥符禅寺", "佛教文化博览馆", "五印坛城"],
    ),
    fact_chunk(
        "fact-niannhua-001",
        "拈花湾禅意小镇点位",
        "route",
        "拈花湾禅意小镇结构化资料包含拈花广场、梵天花海、香月花街、拈花堂、五灯湖、鹿鸣谷六个点位。",
        audience=["all", "photo", "leisure"],
        data_scope="structured_spots",
        keywords=["拈花湾", "拈花广场", "梵天花海", "香月花街", "拈花堂", "五灯湖", "鹿鸣谷"],
    ),
    fact_chunk(
        "fact-open-001",
        "灵山胜境开放时间汇总",
        "opening_hours",
        "开放时间信息：灵山大照壁、五明桥、曼飞龙塔等室外景观点通常全天开放；佛教文化博览馆随灵山大佛开放时间同步运营，8:00-17:00，冬季提前至16:30；五印坛城9:00-17:00开放，冬季提前至16:30。",
        data_scope="partial",
        keywords=["开放时间", "全天开放", "8:00-17:00", "9:00-17:00", "冬季"],
    ),
    fact_chunk(
        "fact-performance-001",
        "灵山胜境演出时间汇总",
        "performance_time",
        "演出时间信息：九龙灌浴平日10:00、11:30、13:30、15:00演出，每场约15分钟；灵山梵宫《灵山吉祥颂》10:35、11:30、14:00、16:00演出，每场约20分钟；节假日可能加演，需以景区广播或官方小程序为准。",
        data_scope="partial",
        keywords=["演出时间", "九龙灌浴", "灵山吉祥颂", "10:00", "10:35", "11:30", "13:30", "14:00", "15:00", "16:00"],
    ),
    fact_chunk(
        "fact-price-001",
        "灵山胜境费用与票价说明",
        "price",
        "费用信息：现有知识库未提供完整景区成人票、儿童票等门票价格表；已知五明桥免费通行，佛教文化博览馆免费参观，灵山梵宫《灵山吉祥颂》凭景区大门票免费入场，导游讲解服务300元起，藏香制作体验费用自理。",
        data_scope="partial",
        keywords=["票价", "费用", "门票", "免费", "导游", "300元起", "费用自理"],
    ),
    fact_chunk(
        "fact-behavior-scope-001",
        "游客行为数据代表性限制",
        "behavior",
        "行为数据需谨慎使用：工作簿包含140447条原始记录、50000名去重游客，灵山或拈花相关记录1613条，其中attraction_name明确命中777条，仅正文弱相关836条；分析灵山胜境游客时应优先使用景点名明确命中的记录，正文弱相关只能作补充参考。",
        audience=["operation", "analytics"],
        data_scope="partial",
        keywords=["行为数据", "代表性", "attraction_name", "明确命中", "正文弱相关", "补充参考"],
    ),
    fact_chunk(
        "fact-behavior-metrics-001",
        "样例游客行为数据指标",
        "behavior",
        "样例游客行为数据可参考年龄、性别、停留时长、消费、团队规模和满意度等聚合指标；当前摘要显示平均停留时长4.23小时，平均团队人数2.62人，平均总消费692.89元，平均满意度3.72。",
        audience=["operation", "analytics"],
        data_scope="partial",
        keywords=["行为数据", "年龄", "性别", "停留时长", "消费", "团队规模", "满意度"],
    ),
]


BASE_QA = [
    ("qa-001", "灵山胜境在哪里？", "灵山胜境坐落于江苏省无锡市太湖西北部的马山镇。", ["fact-location-001", "ls-guide-0001"], "location", ["江苏省无锡市", "太湖西北部", "马山镇"]),
    ("qa-002", "灵山胜境为什么叫小灵山？", "玄奘法师见马山山形酷似印度灵鹫山，赐名小灵山。", ["fact-origin-001", "ls-guide-0002"], "history", ["玄奘", "印度灵鹫山", "小灵山"]),
    ("qa-003", "祥符禅寺的历史可以怎么讲？", "小灵山庵在北宋大中祥符年间获赐额祥符禅寺，历经兴废。", ["fact-history-001", "ls-guide-0003", "ls-spots-0011"], "history", ["小灵山庵", "北宋大中祥符", "祥符禅寺"]),
    ("qa-004", "灵山大佛什么时候落成开光？", "灵山大佛于1997年11月15日落成开光。", ["fact-buddha-001", "ls-guide-0004"], "time", ["1997年11月15日", "落成开光"]),
    ("qa-005", "赵朴初和灵山胜境有什么关系？", "赵朴初提出五方五佛理念，并题写灵山胜境大照壁。", ["fact-zhaopuchu-001", "ls-guide-0004", "ls-guide-0005", "ls-spots-0002"], "history", ["赵朴初", "五方五佛", "题写", "大照壁"]),
    ("qa-006", "灵山梵宫有什么特色？", "灵山梵宫融合佛教艺术、传统工艺和现代科技。", ["fact-fangong-001", "ls-guide-0006", "ls-spots-0014"], "spot_feature", ["佛教艺术", "传统工艺", "现代科技"]),
    ("qa-007", "灵山大照壁适合做什么？", "大照壁是入口标志性门户，适合打卡合影和解读诗刻文化。", ["fact-dazhaobi-001", "ls-spots-0002"], "spot_feature", ["入口", "打卡合影", "诗刻文化"]),
    ("qa-008", "五明桥代表什么含义？", "五明桥象征佛教五种核心智慧。", ["fact-wumingqiao-001", "ls-spots-0003"], "culture", ["五种核心智慧", "声明", "因明", "内明"]),
    ("qa-009", "九龙灌浴的核心看点是什么？", "九龙灌浴以佛祖诞生典故为核心，是重要动态演艺景观。", ["fact-jiulong-001", "ls-guide-0007", "ls-spots-0007"], "performance", ["佛祖诞生", "花开见佛", "九龙沐浴", "动态景观"]),
    ("qa-010", "阿育王柱有什么文化意义？", "阿育王柱体现佛教传播和护法精神。", ["fact-ayuwang-001", "ls-spots-0009"], "culture", ["佛教传播", "和平包容", "护法精神"]),
    ("qa-011", "祥符禅寺在游览中有什么价值？", "祥符禅寺承载千年佛教传承，是灵山胜境核心文化节点。", ["fact-xiangfu-001", "ls-spots-0011", "ls-guide-0003"], "spot_feature", ["千年佛教传承", "核心文化节点", "礼佛祈福"]),
    ("qa-012", "灵山大佛游玩亮点有哪些？", "可瞻仰大佛、登临观景，并感受五方五佛文化格局。", ["fact-buddha-002", "ls-spots-0012"], "spot_feature", ["瞻仰", "观景", "抱佛脚", "五方五佛"]),
    ("qa-013", "佛教文化博览馆适合哪些游客？", "适合想深入了解佛教历史、艺术和文化脉络的游客。", ["fact-museum-001", "ls-spots-0013"], "audience", ["佛教历史", "艺术", "文化脉络", "亲子科普"]),
    ("qa-014", "五印坛城体现什么文化？", "五印坛城体现藏传佛教文化与坛城艺术。", ["fact-wuyin-001", "ls-spots-0016"], "culture", ["藏传佛教", "坛城艺术", "五方五佛"]),
    ("qa-015", "曼飞龙塔有什么特色？", "曼飞龙塔展现南传佛教建筑风格和多元佛教文化。", ["fact-manfeilong-001", "ls-spots-0017"], "culture", ["南传佛教", "白塔", "多元佛教"]),
    ("qa-016", "亲子游客可以关注哪些点？", "可关注九龙灌浴、百子戏弥勒、佛教文化博览馆等互动和文化点位。", ["fact-family-route-001", "ls-guide-0015", "ls-guide-0016"], "route_family", ["九龙灌浴", "百子戏弥勒", "佛教文化博览馆"]),
    ("qa-017", "想拍照打卡怎么安排灵山胜境？", "可优先大照壁、五明桥、九龙灌浴、灵山大佛和梵宫。", ["fact-photo-route-001", "ls-guide-0020"], "route_photo", ["大照壁", "五明桥", "九龙灌浴", "灵山大佛", "梵宫"]),
    ("qa-018", "拈花湾禅意小镇有哪些点位？", "结构化资料包含拈花广场、梵天花海、香月花街、拈花堂、五灯湖、鹿鸣谷。", ["fact-niannhua-001", "ls-spots-0020", "ls-spots-0021", "ls-spots-0022", "ls-spots-0023", "ls-spots-0024", "ls-spots-0025"], "spot_list", ["拈花广场", "梵天花海", "香月花街", "拈花堂", "五灯湖", "鹿鸣谷"]),
    ("qa-019", "行为数据能否代表灵山胜境游客？", "需谨慎；以 attraction_name 明确命中的记录优先，正文弱相关只能作补充参考。", ["fact-behavior-scope-001", "tourism-behavior-0001"], "behavior_scope", ["attraction_name", "明确命中", "正文弱相关", "补充参考"]),
    ("qa-020", "样例游客行为数据能提供什么参考？", "可参考年龄、性别、停留时长、消费、团队规模和满意度等聚合指标。", ["fact-behavior-metrics-001", "tourism-behavior-0002", "tourism-behavior-0003", "tourism-behavior-0004"], "behavior_metrics", ["年龄", "性别", "停留时长", "消费", "团队规模", "满意度"]),
    ("qa-021", "九龙灌浴几点演？", "九龙灌浴平日10:00、11:30、13:30、15:00演出，每场约15分钟。", ["fact-performance-001", "ls-spots-0007"], "performance_time", ["10:00", "11:30", "13:30", "15:00", "15分钟"]),
    ("qa-022", "灵山吉祥颂演出时间是什么？", "《灵山吉祥颂》10:35、11:30、14:00、16:00演出，每场约20分钟。", ["fact-performance-001", "ls-spots-0015"], "performance_time", ["10:35", "11:30", "14:00", "16:00", "20分钟"]),
    ("qa-023", "佛教文化博览馆开放到几点？", "佛教文化博览馆随灵山大佛开放时间同步运营，8:00-17:00，冬季提前至16:30。", ["fact-open-001", "ls-spots-0013"], "opening_hours", ["8:00-17:00", "冬季", "16:30"]),
    ("qa-024", "五印坛城开放时间？", "五印坛城9:00-17:00开放，冬季闭馆时间提前至16:30。", ["fact-open-001", "ls-spots-0016"], "opening_hours", ["9:00-17:00", "冬季", "16:30"]),
    ("qa-025", "灵山胜境门票多少钱？", "现有知识库没有完整门票价格表；只记录部分免费项目、导游300元起和部分体验费用自理。", ["fact-price-001"], "price_limit", ["未提供完整门票价格表", "免费项目", "300元起", "费用自理"]),
]


PARAPHRASES = [
    ("灵山胜境具体在无锡哪里？", "location"),
    ("小灵山这个名字是谁起的？", "history"),
    ("祥符禅寺为什么说有千年历史？", "history"),
    ("灵山大佛是哪一天开光的？", "time"),
    ("赵朴初为灵山做过什么？", "history"),
    ("梵宫为什么值得看？", "spot_feature"),
    ("大照壁是不是适合拍照？", "spot_feature"),
    ("五明桥的五明指什么？", "culture"),
    ("九龙灌浴主要表演什么故事？", "performance"),
    ("阿育王柱和佛教传播有什么关系？", "culture"),
    ("祥符禅寺适合祈福吗？", "spot_feature"),
    ("灵山大佛有哪些体验点？", "spot_feature"),
    ("带孩子去佛教文化博览馆合适吗？", "audience"),
    ("五印坛城是不是藏传佛教建筑？", "culture"),
    ("曼飞龙塔属于哪种佛教风格？", "culture"),
    ("亲子游灵山先看哪些景点？", "route_family"),
    ("灵山拍照路线怎么排？", "route_photo"),
    ("拈花湾有哪些主要景点？", "spot_list"),
    ("游客行为数据可以直接代表灵山吗？", "behavior_scope"),
    ("游客行为摘要有哪些指标？", "behavior_metrics"),
    ("九龙灌浴平日表演时间？", "performance_time"),
    ("吉祥颂每天哪几个时间段？", "performance_time"),
    ("博览馆冬季几点闭馆？", "opening_hours"),
    ("五印坛城冬天开放到几点？", "opening_hours"),
    ("知识库里有完整票价吗？", "price_limit"),
    ("灵山胜景在哪里？", "location"),
    ("玄奘为什么叫这里小灵山？", "history"),
    ("祥符禅寺北宋时发生了什么？", "history"),
    ("灵山大佛1997年发生什么？", "time"),
    ("赵朴初题写了哪里？", "history"),
    ("灵山梵官有什么特色？", "spot_feature"),
    ("灵山大昭壁能不能打卡？", "spot_feature"),
    ("五名桥有什么寓意？", "culture"),
    ("九龙灌浴看点是喷泉吗？", "performance"),
    ("阿育王住象征什么？", "culture"),
    ("祥福禅寺值不值得看？", "spot_feature"),
    ("大佛那里怎么玩？", "spot_feature"),
    ("博览馆适合研学游客吗？", "audience"),
    ("五印坛城体现什么佛教语系？", "culture"),
    ("曼飞龙塔拍照有什么特色？", "culture"),
    ("带娃去灵山看什么？", "route_family"),
    ("打卡照优先去哪几个点？", "route_photo"),
    ("拈花湾六个点位有哪些？", "spot_list"),
    ("行为表里正文弱相关能当主数据吗？", "behavior_scope"),
    ("样例行为数据有满意度吗？", "behavior_metrics"),
    ("九龙灌浴下午几点有？", "performance_time"),
    ("灵山吉祥颂一场多久？", "performance_time"),
    ("佛教文化博览馆要另外收费吗？", "price_limit"),
    ("五明桥要门票吗？", "price_limit"),
    ("如果问成人票价格能直接回答吗？", "price_limit"),
    ("灵山胜境不在无锡吗？", "location"),
    ("不是玄奘给小灵山命名的吗？", "history"),
    ("祥符禅寺难道不是北宋赐额？", "history"),
    ("灵山大佛不是1997年开光的吗？", "time"),
    ("赵朴初没有参与灵山大佛文化格局吗？", "history"),
    ("灵山梵宫是不是只是一座普通建筑？", "spot_feature"),
    ("大照壁除了路过还能看什么？", "spot_feature"),
    ("五明桥不是五座桥而已吗？", "culture"),
    ("九龙灌浴是不是只适合拍照？", "performance"),
    ("阿育王柱不就是柱子吗，有文化含义吗？", "culture"),
    ("祥符禅寺只是普通寺庙吗？", "spot_feature"),
    ("灵山大佛除了看佛像还有什么？", "spot_feature"),
    ("不了解佛教的人适合去博览馆吗？", "audience"),
    ("五印坛城和藏传佛教有关吗？", "culture"),
    ("曼飞龙塔是不是南传风格？", "culture"),
    ("亲子游不看演出可以吗，推荐哪些？", "route_family"),
    ("拍照不去梵宫可以吗，路线怎么补？", "route_photo"),
    ("拈花湾是不是只有拈花广场？", "spot_list"),
    ("行为数据是不是全部都是灵山游客？", "behavior_scope"),
    ("样例数据能看出平均消费吗？", "behavior_metrics"),
    ("九龍灌浴什么时候表演？", "performance_time"),
    ("灵山吉详颂几点开始？", "performance_time"),
    ("博览馆开放时段是全天吗？", "opening_hours"),
    ("五印坛城是不是晚上也开放？", "opening_hours"),
    ("门票价目表在知识库里完整吗？", "price_limit"),
]


OUT_OF_SCOPE = [
    ("qa-oos-001", "灵山胜境今天实时客流是多少？", "现有知识库没有实时客流数据，不能编造。", [], "out_of_scope", ["没有实时客流数据"]),
    ("qa-oos-002", "明天九龙灌浴会不会临时取消？", "现有知识库只有常规演出时间，临时调整需以景区广播或官方小程序为准。", ["fact-performance-001"], "out_of_scope", ["临时调整", "官方小程序"]),
    ("qa-oos-003", "灵山胜境2026年成人票最新价格是多少？", "现有知识库未提供完整最新票价表，不能确认成人票最新价格。", ["fact-price-001"], "out_of_scope", ["未提供完整最新票价表"]),
    ("qa-oos-004", "灵山附近哪家酒店最便宜？", "现有知识库不包含周边酒店价格数据。", [], "out_of_scope", ["不包含酒店价格"]),
    ("qa-oos-005", "帮我买两张明天的门票。", "系统当前知识库不能代购门票，也没有实时售票接口。", ["fact-price-001"], "out_of_scope", ["不能代购门票"]),
    ("qa-oos-006", "灵山大佛现在排队多久？", "现有知识库没有实时排队时长。", [], "out_of_scope", ["没有实时排队时长"]),
    ("qa-oos-007", "今天无锡天气适合去灵山吗？", "现有知识库不包含实时天气。", [], "out_of_scope", ["不包含实时天气"]),
    ("qa-oos-008", "景区客服电话是多少？", "现有知识库没有客服电话。", [], "out_of_scope", ["没有客服电话"]),
    ("qa-oos-009", "灵山胜境有没有最新优惠券？", "现有知识库没有实时优惠券信息。", [], "out_of_scope", ["没有实时优惠券"]),
    ("qa-oos-010", "给我预测明年五一游客量。", "现有知识库不足以预测未来五一游客量。", ["fact-behavior-scope-001"], "out_of_scope", ["不足以预测"]),
    ("qa-oos-011", "灵山胜境停车场现在还有空位吗？", "现有知识库没有实时停车位数据。", [], "out_of_scope", ["没有实时停车位数据"]),
    ("qa-oos-012", "梵宫今天哪位演员上场？", "现有知识库没有演职人员实时排班。", ["fact-performance-001"], "out_of_scope", ["没有演职人员实时排班"]),
    ("qa-oos-013", "帮我预约明天藏香制作。", "现有知识库只说明藏香制作需预约，不能代办预约。", ["fact-open-001"], "out_of_scope", ["不能代办预约"]),
    ("qa-oos-014", "灵山胜境附近打车多少钱？", "现有知识库没有实时打车价格。", [], "out_of_scope", ["没有实时打车价格"]),
    ("qa-oos-015", "拈花湾今晚灯光秀准确时间？", "现有知识库没有拈花湾实时演艺排期。", ["fact-niannhua-001"], "out_of_scope", ["没有实时演艺排期"]),
]


def build_qa() -> list[dict[str, Any]]:
    qa_by_id: dict[str, dict[str, Any]] = {}
    base_by_type: dict[str, tuple[str, str, list[str], list[str]]] = {}
    for case_id, question, hint, chunk_ids, question_type, facts in BASE_QA:
        qa_by_id[case_id] = {
            "id": case_id,
            "question": question,
            "expected_answer_hint": hint,
            "expected_chunk_ids": chunk_ids,
            "question_type": question_type,
            "key_facts": facts,
            "source": "data/knowledge_base/chunks/chunks.jsonl",
        }
        base_by_type.setdefault(question_type, (hint, chunk_ids, facts, [case_id]))

    def infer_base_id(question: str, fallback_type: str) -> str:
        rules = [
            ("赵朴初", "qa-005"),
            ("胜景", "qa-001"),
            ("位置", "qa-001"),
            ("无锡", "qa-001"),
            ("哪里", "qa-001"),
            ("小灵山", "qa-002"),
            ("玄奘", "qa-002"),
            ("北宋", "qa-003"),
            ("赐额", "qa-003"),
            ("千年历史", "qa-003"),
            ("1997", "qa-004"),
            ("开光", "qa-004"),
            ("拍照不去梵宫", "qa-017"),
            ("吉祥颂", "qa-022"),
            ("吉详颂", "qa-022"),
            ("梵宫", "qa-006"),
            ("梵官", "qa-006"),
            ("大照壁", "qa-007"),
            ("大昭壁", "qa-007"),
            ("五明桥", "qa-008"),
            ("五名桥", "qa-008"),
            ("九龙灌浴下午", "qa-021"),
            ("九龙灌浴平日", "qa-021"),
            ("九龍灌浴什么时候", "qa-021"),
            ("九龙灌浴几点", "qa-021"),
            ("九龙灌浴", "qa-009"),
            ("阿育王柱", "qa-010"),
            ("阿育王住", "qa-010"),
            ("祥符禅寺适合", "qa-011"),
            ("祥福禅寺", "qa-011"),
            ("普通寺庙", "qa-011"),
            ("灵山大佛有哪些体验", "qa-012"),
            ("大佛那里怎么玩", "qa-012"),
            ("除了看佛像", "qa-012"),
            ("博览馆冬季", "qa-023"),
            ("博览馆开放", "qa-023"),
            ("佛教文化博览馆要另外收费", "qa-025"),
            ("博览馆适合", "qa-013"),
            ("佛教文化博览馆", "qa-013"),
            ("五印坛城冬", "qa-024"),
            ("五印坛城晚上", "qa-024"),
            ("五印坛城", "qa-014"),
            ("曼飞龙塔", "qa-015"),
            ("亲子", "qa-016"),
            ("带孩子", "qa-016"),
            ("带娃", "qa-016"),
            ("拍照", "qa-017"),
            ("打卡", "qa-017"),
            ("拈花湾", "qa-018"),
            ("行为", "qa-019"),
            ("正文弱相关", "qa-019"),
            ("attraction_name", "qa-019"),
            ("满意度", "qa-020"),
            ("平均消费", "qa-020"),
            ("行为摘要", "qa-020"),
            ("票", "qa-025"),
            ("门票", "qa-025"),
            ("收费", "qa-025"),
            ("价格", "qa-025"),
            ("价目表", "qa-025"),
        ]
        for keyword, case_id in rules:
            if keyword in question:
                return case_id
        for item in qa_by_id.values():
            if item["question_type"] == fallback_type:
                return item["id"]
        return "qa-001"

    index = 26
    for question, question_type in PARAPHRASES:
        reference = qa_by_id[infer_base_id(question, question_type)]
        key_facts = reference["key_facts"]
        if reference["question_type"] == "price_limit" and any(
            word in question for word in ["完整", "成人票", "价目表", "价格", "多少钱"]
        ):
            key_facts = ["未提供完整门票价格表"]
        qa_by_id[f"qa-{index:03d}"] = {
            "id": f"qa-{index:03d}",
            "question": question,
            "expected_answer_hint": reference["expected_answer_hint"],
            "expected_chunk_ids": reference["expected_chunk_ids"],
            "question_type": reference["question_type"],
            "key_facts": key_facts,
            "source": "data/knowledge_base/chunks/chunks.jsonl",
        }
        index += 1

    for case_id, question, hint, chunk_ids, question_type, facts in OUT_OF_SCOPE:
        qa_by_id[case_id] = {
            "id": case_id,
            "question": question,
            "expected_answer_hint": hint,
            "expected_chunk_ids": chunk_ids,
            "question_type": question_type,
            "key_facts": facts,
            "source": "data/knowledge_base/chunks/chunks.jsonl",
        }

    return list(qa_by_id.values())


def main() -> None:
    original = load_jsonl(CHUNKS_FILE)
    base_chunks = [row for row in original if row.get("document_id") != "ling_shan_curated_facts"]
    chunks = [enrich_chunk(row) for row in base_chunks] + FACTS
    write_jsonl(CHUNKS_FILE, chunks)

    qa = build_qa()
    QA_FILE.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    counts = Counter(row.get("document_id", "") for row in chunks)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["chunk_count"] = len(chunks)
    manifest["qa_count"] = len(qa)
    manifest["chunks_by_document"] = dict(counts)
    notes = manifest.setdefault("notes", [])
    new_note = "Curated fact chunks split opening hours, performance times, price scope, routes, audiences, and behavior-data limits for RAG evaluation."
    if new_note not in notes:
        notes.append(new_note)
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "chunks": len(chunks),
                "curated_facts": len(FACTS),
                "qa_cases": len(qa),
                "qa_by_type": dict(Counter(item["question_type"] for item in qa)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
