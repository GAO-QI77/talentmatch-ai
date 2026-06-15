from __future__ import annotations

import csv
import hashlib
import io
import re
from datetime import datetime
from typing import Any, Iterable


CRM_STATUSES = [
    "未触达",
    "已生成话术",
    "待人工确认",
    "已联系",
    "已回复",
    "感兴趣",
    "不感兴趣",
    "进入面试",
    "已淘汰",
    "加入人才池",
    "不再联系",
    "黑名单",
]

SOURCE_TYPES = [
    "manual_input",
    "csv_upload",
    "authorized_talent_pool",
    "public_profile",
    "mock_data",
    "mock_search",
    "local_talent_pool",
    "search_api",
    "ats",
    "official_api_placeholder",
]

RECOMMENDATION_LEVELS = ["强烈推荐", "可以联系", "观望", "不推荐"]
DAILY_OUTREACH_LIMIT = 20
NON_CONTACTABLE_STATUSES = ["黑名单", "不再联系"]

SENSITIVE_TERM_MAP = {
    "年龄": ["年龄", "岁", "出生", "属相"],
    "婚育": ["已婚", "未婚", "婚育", "结婚", "孩子", "生育"],
    "健康": ["健康", "病史", "残疾", "怀孕", "体检"],
    "政治": ["党员", "党派", "政治", "团员"],
    "宗教": ["宗教", "信仰", "佛教", "基督", "伊斯兰"],
    "民族": ["民族", "汉族", "回族", "藏族", "维吾尔"],
}

REQUIRED_CANDIDATE_FIELDS = [
    "candidate_id",
    "name",
    "current_company",
    "current_title",
    "location",
    "years_of_experience",
    "skills",
    "education",
    "work_experience",
    "project_experience",
    "public_profile_text",
    "profile_url",
    "source_type",
    "source_note",
    "last_updated",
    "contact_status",
    "notes",
    "blacklist_status",
    "retrieval_source",
    "retrieval_query",
    "retrieved_at",
    "source_confidence",
    "profile_completeness",
    "sourcing_status",
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip().lower())


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}{digest}"


def recommendation_from_score(score: float | int) -> str:
    score = float(score or 0)
    if score >= 85:
        return "强烈推荐"
    if score >= 70:
        return "可以联系"
    if score >= 45:
        return "观望"
    return "不推荐"


def detect_sensitive_terms(text: str) -> list[str]:
    findings: list[str] = []
    normalized = normalize_text(text)
    for category, terms in SENSITIVE_TERM_MAP.items():
        if any(term.lower() in normalized for term in terms):
            findings.append(category)
    return findings


def contactable_reason(candidate: dict[str, Any]) -> tuple[bool, str]:
    if candidate.get("blacklist_status") == "是" or candidate.get("contact_status") == "黑名单":
        return False, "候选人已在黑名单，禁止生成触达。"
    if candidate.get("contact_status") == "不再联系":
        return False, "候选人已标记不再联系，禁止生成触达。"
    return True, "可生成草稿，但必须由 HR 人工确认后再复制到合规渠道。"


def next_best_action(candidate: dict[str, Any], match_result: dict[str, Any] | None = None) -> str:
    match_result = match_result or {}
    contactable, _ = contactable_reason(candidate)
    if not contactable:
        return "不再联系"
    if not match_result:
        return "开始匹配"
    if not normalize_text(candidate.get("skills")) or normalize_text(candidate.get("current_title")) in ["", "未明确"]:
        return "补充资料"
    status = candidate.get("contact_status", "未触达")
    if status in ["已生成话术", "待人工确认"]:
        return "人工确认"
    if status in ["已回复", "感兴趣"]:
        return "安排面试"
    if status in ["已联系", "进入面试", "加入人才池"]:
        return "CRM 跟进"
    score = int(match_result.get("match_score", 0) or 0)
    if score >= 70:
        return "生成草稿"
    return "人工复核"


def build_compliance_status(
    candidate: dict[str, Any],
    outreach_used: int,
    daily_limit: int = DAILY_OUTREACH_LIMIT,
) -> dict[str, Any]:
    sensitive_findings = detect_sensitive_terms(
        " ".join(
            [
                str(candidate.get("public_profile_text", "")),
                str(candidate.get("notes", "")),
                str(candidate.get("work_experience", "")),
            ]
        )
    )
    contactable, reason = contactable_reason(candidate)
    remaining = max(0, daily_limit - int(outreach_used or 0))
    return {
        "data_source": candidate.get("source_type", "manual_input"),
        "sensitive_findings": sensitive_findings,
        "contactable": bool(contactable and remaining > 0),
        "contact_guardrail": reason if remaining > 0 else "今日建议触达额度已用完，请降低触达频率。",
        "remaining_quota": remaining,
        "manual_confirmation_required": True,
    }


def _candidate_evidence_text(candidate: dict[str, Any], requirement: str) -> str:
    haystacks = [
        candidate.get("skills", ""),
        candidate.get("work_experience", ""),
        candidate.get("project_experience", ""),
        candidate.get("public_profile_text", ""),
    ]
    keywords = [w for w in re.split(r"[\s,/，、；;]+", str(requirement).lower()) if len(w) >= 2]
    for text in haystacks:
        normalized = normalize_text(text)
        if keywords and any(word in normalized for word in keywords):
            return str(text).strip()[:90]
    return "候选人资料中缺少直接证据，需人工补充核实。"


def build_match_evidence(job_profile: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, str]]:
    requirements = list(job_profile.get("must_have_requirements") or [])[:3]
    requirements += list(job_profile.get("nice_to_have_requirements") or [])[:2]
    evidence = []
    for requirement in requirements:
        evidence.append(
            {
                "job_requirement": str(requirement),
                "candidate_evidence": _candidate_evidence_text(candidate, str(requirement)),
            }
        )
    if not evidence:
        evidence.append(
            {
                "job_requirement": "岗位要求未明确",
                "candidate_evidence": "信息不足，无法判断。请先完善 JD 或候选人资料。",
            }
        )
    return evidence


def confidence_from_missing_info(missing_information: list[str], evidence: list[dict[str, str]]) -> str:
    missing_count = len([item for item in missing_information if "不足" in item or "不明确" in item])
    weak_evidence = len([item for item in evidence if "缺少直接证据" in item.get("candidate_evidence", "")])
    if missing_count == 0 and weak_evidence <= 1:
        return "高"
    if missing_count <= 2 and weak_evidence <= 3:
        return "中"
    return "低"


def ensure_candidate_defaults(candidate: dict[str, Any]) -> dict[str, Any]:
    merged = {field: "" for field in REQUIRED_CANDIDATE_FIELDS}
    merged.update(candidate)
    seed = "|".join(
        [
            normalize_text(merged.get("name")),
            normalize_text(merged.get("current_company")),
            normalize_text(merged.get("current_title")),
            normalize_text(merged.get("public_profile_text")),
        ]
    )
    if not normalize_text(merged.get("candidate_id")):
        merged["candidate_id"] = stable_id("C", seed or now_text())
    if not normalize_text(merged.get("source_type")):
        merged["source_type"] = "manual_input"
    if not normalize_text(merged.get("last_updated")):
        merged["last_updated"] = now_text()
    if not normalize_text(merged.get("contact_status")):
        merged["contact_status"] = "未触达"
    if not normalize_text(merged.get("blacklist_status")):
        merged["blacklist_status"] = "否"
    if not normalize_text(merged.get("retrieval_source")):
        merged["retrieval_source"] = merged.get("source_type") or "manual_input"
    if not normalize_text(merged.get("retrieved_at")):
        merged["retrieved_at"] = ""
    if not normalize_text(merged.get("source_confidence")):
        merged["source_confidence"] = "中"
    if not normalize_text(merged.get("profile_completeness")):
        merged["profile_completeness"] = 0
    if not normalize_text(merged.get("sourcing_status")):
        merged["sourcing_status"] = "已入库" if merged.get("source_type") != "mock_search" else "待复核"
    return merged


def build_candidate_from_profile_text(profile_text: str) -> dict[str, Any]:
    seed = normalize_text(profile_text)
    candidate = {
        "candidate_id": stable_id("M", seed or now_text()),
        "name": "待补充候选人",
        "current_company": "未明确",
        "current_title": "未明确",
        "location": "未明确",
        "years_of_experience": "未明确",
        "skills": "",
        "education": "未明确",
        "work_experience": "",
        "project_experience": "",
        "public_profile_text": profile_text.strip(),
        "profile_url": "",
        "source_type": "manual_input",
        "source_note": "由招聘人员手动粘贴的候选人主页文本",
        "contact_status": "未触达",
        "notes": "",
        "blacklist_status": "否",
    }
    return ensure_candidate_defaults(candidate)


def candidate_identity_keys(candidate: dict[str, Any]) -> list[tuple[str, str]]:
    normalized = ensure_candidate_defaults(candidate)
    keys: list[tuple[str, str]] = []
    candidate_id = normalize_text(normalized.get("candidate_id"))
    profile_url = normalize_text(normalized.get("profile_url"))
    composite = "|".join(
        [
            normalize_text(normalized.get("name")),
            normalize_text(normalized.get("current_company")),
            normalize_text(normalized.get("current_title")),
        ]
    )
    if candidate_id:
        keys.append(("candidate_id", candidate_id))
    if profile_url:
        keys.append(("profile_url", profile_url))
    if composite.replace("|", ""):
        keys.append(("name_company_title", composite))
    return keys


def dedupe_candidates(
    candidates: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    seen: dict[tuple[str, str], str] = {}
    unique: list[dict[str, Any]] = []
    warnings: list[str] = []

    for raw in candidates:
        candidate = ensure_candidate_defaults(dict(raw))
        duplicate_fields: list[str] = []
        duplicate_of = ""
        for key in candidate_identity_keys(candidate):
            if key in seen:
                duplicate_fields.append(key[0])
                duplicate_of = seen[key]

        if duplicate_fields:
            fields = " / ".join(sorted(set(duplicate_fields)))
            warnings.append(
                f"候选人 {candidate.get('candidate_id')} 与 {duplicate_of} 可能重复，命中字段：{fields}"
            )
            continue

        unique.append(candidate)
        for key in candidate_identity_keys(candidate):
            seen[key] = str(candidate.get("candidate_id"))

    return unique, warnings


def add_audit_log(logs: list[dict[str, Any]], action: str, detail: str) -> None:
    logs.insert(
        0,
        {
            "time": now_text(),
            "action": action,
            "detail": detail,
        },
    )


def candidate_search_text(candidate: dict[str, Any]) -> str:
    fields = [
        "name",
        "current_company",
        "current_title",
        "location",
        "skills",
        "education",
        "work_experience",
        "project_experience",
        "public_profile_text",
    ]
    return " ".join(str(candidate.get(field, "")) for field in fields)


def calculate_mock_score(job_profile: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(candidate_search_text(candidate))
    must_haves = job_profile.get("must_have_requirements") or []
    nice_haves = job_profile.get("nice_to_have_requirements") or []
    responsibilities = job_profile.get("responsibilities") or []
    selling_points = job_profile.get("company_selling_points") or []

    def hit_count(items: Iterable[Any]) -> int:
        count = 0
        for item in items:
            words = [w for w in re.split(r"[\s,/，、；;]+", str(item).lower()) if len(w) >= 2]
            if any(word in text for word in words):
                count += 1
        return count

    must_score = min(35, 12 + hit_count(must_haves) * 8)
    industry_score = 8
    if any(token in text for token in ["hr", "招聘", "人才", "saa", "saas", "猎头"]):
        industry_score += 8
    if any(token in text for token in ["ai", "大模型", "llm", "智能"]):
        industry_score += 4
    skills_score = min(25, 8 + hit_count(nice_haves + responsibilities) * 4)
    career_score = 6 if any(token in text for token in ["产品", "负责人", "经理", "lead"]) else 4
    outreach_score = 6 + min(4, hit_count(selling_points))
    total = int(min(100, must_score + industry_score + skills_score + career_score + outreach_score))

    reasons = []
    if "ai" in text or "大模型" in text:
        reasons.append("资料中出现 AI / 大模型相关经历，可支撑 AI 招聘工具方向。")
    if "hr" in text or "招聘" in text or "人才" in text:
        reasons.append("候选人资料与 HR、招聘或人才业务存在明确交集。")
    if "streamlit" in text or "python" in text or "数据" in text:
        reasons.append("具备 Python、数据处理或 Demo 搭建相关线索。")
    if not reasons:
        reasons.append("候选人资料较少，需要人工进一步核实与岗位的直接相关性。")

    risks = []
    if "saas" not in text and "saa" not in text:
        risks.append("缺少 B2B SaaS 或 HR SaaS 经验证据。")
    if "ai" not in text and "大模型" not in text:
        risks.append("AI 产品或大模型调用经验不明确。")
    if normalize_text(candidate.get("location")) in ["", "未明确"]:
        risks.append("地点匹配不明确。")

    missing = []
    for label, field in [
        ("教育背景", "education"),
        ("项目经历", "project_experience"),
        ("技能栈", "skills"),
    ]:
        if normalize_text(candidate.get(field)) in ["", "未明确"]:
            missing.append(f"{label}信息不足，无法判断。")

    evidence = build_match_evidence(job_profile, candidate)
    confidence = confidence_from_missing_info(missing, evidence)

    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "candidate_name": candidate.get("name", ""),
        "match_score": total,
        "recommendation_level": recommendation_from_score(total),
        "confidence_level": confidence,
        "evidence": evidence,
        "score_breakdown": {
            "must_have": int(must_score),
            "industry_background": int(industry_score),
            "skills_projects": int(skills_score),
            "career_path": int(career_score),
            "outreach_priority": int(outreach_score),
        },
        "match_reasons": reasons,
        "risks": risks,
        "missing_information": missing or ["暂无明显缺失，但仍建议人工复核。"],
        "interview_focus": [
            "请候选人展开说明最近一次与岗位相关的项目。",
            "确认 AI 工具落地、数据处理和跨团队协作的真实深度。",
        ],
        "outreach_angle": reasons[0],
        "suggested_next_action": "生成触达草稿后由 HR manual confirmation 人工确认，再复制到官方 App 或授权渠道。",
    }


def parse_csv_bytes(data: bytes) -> list[dict[str, Any]]:
    text = data.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    return [ensure_candidate_defaults(row) for row in rows]


def dataframe_for_export(candidates: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=REQUIRED_CANDIDATE_FIELDS)
    writer.writeheader()
    for candidate in candidates:
        row = ensure_candidate_defaults(candidate)
        writer.writerow({field: row.get(field, "") for field in REQUIRED_CANDIDATE_FIELDS})
    return output.getvalue()
