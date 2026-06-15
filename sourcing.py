from __future__ import annotations

import re
from typing import Any, Iterable

from utils import (
    candidate_search_text,
    ensure_candidate_defaults,
    normalize_text,
    now_text,
    stable_id,
)


SOURCE_CONFIDENCE_BY_TYPE = {
    "mock_search": "中",
    "local_talent_pool": "高",
    "search_api": "中",
    "ats": "高",
    "official_api_placeholder": "高",
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _split_terms(value: Any) -> list[str]:
    terms: list[str] = []
    for item in _as_list(value):
        parts = re.split(r"[\s,/，、；;|]+", item)
        for part in parts:
            clean = part.strip()
            if len(clean) >= 2 and clean not in terms:
                terms.append(clean)
    return terms


def _dedupe(items: Iterable[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = str(item).strip()
        key = normalize_text(clean)
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
        if limit and len(result) >= limit:
            break
    return result


def build_search_strategy(job_profile: dict[str, Any]) -> dict[str, Any]:
    role_title = str(job_profile.get("role_title") or "AI 招聘工具产品经理").strip()
    locations = _dedupe(_split_terms(job_profile.get("location")) or ["远程"], limit=4)
    must_phrases = _as_list(job_profile.get("must_have_requirements"))
    nice_phrases = _as_list(job_profile.get("nice_to_have_requirements"))
    responsibility_phrases = _as_list(job_profile.get("responsibilities"))
    must_terms = _split_terms(job_profile.get("must_have_requirements"))
    nice_terms = _split_terms(job_profile.get("nice_to_have_requirements"))
    responsibilities = _split_terms(job_profile.get("responsibilities"))
    excluded = _dedupe(_split_terms(job_profile.get("excluded_conditions")) or ["纯销售", "无产品经验"], limit=6)
    keywords = _dedupe(
        must_phrases + nice_phrases + responsibility_phrases + must_terms + nice_terms + responsibilities,
        limit=12,
    )
    title_aliases = _dedupe(
        [
            role_title,
            role_title.replace("AI-HR", "AI 招聘"),
            "AI 产品经理",
            "HR SaaS 产品经理",
            "招聘产品经理",
            "人才产品经理",
            "招聘运营产品",
        ],
        limit=7,
    )
    industries = _dedupe(
        [
            "HR SaaS",
            "AI",
            "招聘科技",
            "企业服务",
            "B2B SaaS",
        ],
        limit=5,
    )
    target_companies = _dedupe(
        [
            "北森",
            "Moka",
            "Boss 直聘",
            "脉脉",
            "LinkedIn",
            "钉钉",
        ],
        limit=6,
    )

    core_terms = keywords[:5] or ["AI 产品", "招聘", "SaaS"]
    boolean_queries = []
    for alias in title_aliases[:4]:
        positive = " AND ".join(_dedupe([alias] + core_terms[:3]))
        negative = " ".join(f"NOT {term}" for term in excluded[:2])
        query = f"({positive}) {negative}".strip()
        boolean_queries.append(query)
    for location in locations[:2]:
        boolean_queries.append(f"({role_title} AND {' AND '.join(core_terms[:2])}) AND {location}")
    boolean_queries.append(f"({' OR '.join(title_aliases[:3])}) AND ({' OR '.join(industries[:3])})")

    return {
        "keywords": keywords,
        "excluded_keywords": excluded,
        "locations": locations,
        "title_aliases": title_aliases,
        "target_companies": target_companies,
        "industries": industries,
        "boolean_queries": _dedupe(boolean_queries, limit=8),
    }


def profile_completeness(candidate: dict[str, Any]) -> int:
    weights = {
        "name": 8,
        "current_company": 10,
        "current_title": 12,
        "location": 8,
        "years_of_experience": 8,
        "skills": 14,
        "education": 8,
        "work_experience": 12,
        "project_experience": 10,
        "public_profile_text": 6,
        "profile_url": 4,
    }
    total = 0
    for field, weight in weights.items():
        value = normalize_text(candidate.get(field))
        if value and value not in ["未明确", "nan", "none"]:
            total += weight
    return min(100, total)


def _hit_score(strategy: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, list[str]]:
    text = normalize_text(candidate_search_text(candidate))
    hits: list[str] = []
    for term in strategy.get("keywords", []) + strategy.get("title_aliases", []) + strategy.get("locations", []):
        clean = normalize_text(term)
        if clean and clean in text:
            hits.append(str(term))
    for term in strategy.get("excluded_keywords", []):
        clean = normalize_text(term)
        if clean and clean in text:
            return 0, []
    return len(set(hits)), _dedupe(hits, limit=8)


def run_mock_search(
    strategy: dict[str, Any],
    pool: list[dict[str, Any]],
    limit: int = 8,
    source_type: str = "mock_search",
) -> list[dict[str, Any]]:
    leads: list[dict[str, Any]] = []
    queries = strategy.get("boolean_queries") or ["AI 产品 AND 招聘"]
    for candidate in pool:
        score, hits = _hit_score(strategy, candidate)
        if score <= 0:
            continue
        query = queries[min(score - 1, len(queries) - 1)]
        lead = dict(candidate)
        lead.update(
            {
                "candidate_id": candidate.get("candidate_id") or stable_id("S", candidate_search_text(candidate)),
                "retrieval_source": source_type,
                "retrieval_query": query,
                "retrieved_at": now_text(),
                "source_confidence": SOURCE_CONFIDENCE_BY_TYPE.get(source_type, "中"),
                "profile_completeness": profile_completeness(candidate),
                "sourcing_status": "新发现",
                "source_type": source_type,
                "source_note": f"自动检索命中：{'、'.join(hits) or '关键词匹配'}",
            }
        )
        leads.append(lead)
    leads.sort(
        key=lambda item: (
            int(item.get("profile_completeness", 0) or 0),
            len(str(item.get("source_note", ""))),
        ),
        reverse=True,
    )
    return [ensure_candidate_defaults(item) for item in leads[:limit]]


def candidate_from_sourcing_result(lead: dict[str, Any]) -> dict[str, Any]:
    source = lead.get("retrieval_source") or lead.get("source_type") or "mock_search"
    candidate = dict(lead)
    candidate.update(
        {
            "source_type": source,
            "retrieval_source": source,
            "retrieved_at": lead.get("retrieved_at") or now_text(),
            "source_confidence": lead.get("source_confidence") or SOURCE_CONFIDENCE_BY_TYPE.get(source, "中"),
            "profile_completeness": lead.get("profile_completeness") or profile_completeness(lead),
            "sourcing_status": "待复核",
            "contact_status": lead.get("contact_status") or "未触达",
            "blacklist_status": lead.get("blacklist_status") or "否",
            "source_note": lead.get("source_note") or "自动检索线索，等待 HR 复核。",
        }
    )
    return ensure_candidate_defaults(candidate)
