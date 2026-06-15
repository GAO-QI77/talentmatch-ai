from __future__ import annotations

import json
import re
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


ACCESS_LIMITED_STATUSES = {"login_required", "captcha", "blocked", "app_only", "permission_required"}
SCORE_WEIGHTS = {
    "name": 40,
    "title": 20,
    "school": 20,
    "company": 15,
    "extra_keywords": 5,
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value).strip())


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", "", _clean(value).lower())


def _as_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"[\s,/，、；;|]+", str(value))
    terms: list[str] = []
    seen: set[str] = set()
    for item in items:
        term = _clean(item)
        key = _normalize(term)
        if len(term) >= 2 and key and key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = _clean(item)
        key = _normalize(clean)
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def build_maimai_queries(input_data: dict[str, Any]) -> list[str]:
    name = _clean(input_data.get("candidate_name"))
    if not name:
        raise ValueError("candidate_name is required")

    queries = [
        f"{name} 脉脉",
        f"site:maimai.cn {name}",
    ]
    for field in ["known_title", "known_company", "known_school"]:
        value = _clean(input_data.get(field))
        if value:
            queries.append(f"{name} {value} 脉脉")
    return _dedupe(queries)


def _is_maimai_related(item: dict[str, str]) -> bool:
    url = _normalize(item.get("url"))
    text = _normalize(f"{item.get('title', '')} {item.get('snippet', '')}")
    return "maimai.cn" in url or "脉脉" in text


def normalize_search_results(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for raw in results:
        item = {
            "title": _clean(raw.get("title") or raw.get("name")),
            "snippet": _clean(raw.get("snippet") or raw.get("description")),
            "url": _clean(raw.get("url") or raw.get("link")),
            "source_query": _clean(raw.get("source_query") or raw.get("query")),
            "access_status": _clean(raw.get("access_status") or raw.get("status") or "search_snippet_only"),
        }
        if not item["url"] or not _is_maimai_related(item):
            continue
        key = _canonical_url(item["url"]) or item["url"]
        if key in seen_urls:
            continue
        seen_urls.add(key)
        normalized.append(item)
    return normalized


def parse_search_results_json(raw_json: str) -> tuple[list[dict[str, Any]], str]:
    raw = _clean(raw_json)
    if not raw:
        return [], ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], f"搜索结果 JSON 解析失败：{exc.msg}"
    if isinstance(parsed, dict):
        parsed = parsed.get("results") or parsed.get("items") or []
    if not isinstance(parsed, list):
        return [], "搜索结果 JSON 必须是数组，或包含 results/items 数组字段。"
    results = [item for item in parsed if isinstance(item, dict)]
    return results, ""


def _canonical_url(url: str) -> str:
    clean = _clean(url)
    if not clean:
        return ""
    parts = urlsplit(clean)
    if not parts.scheme or not parts.netloc:
        return clean
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") or "/", "", ""))


def _first_candidate_name(text: str) -> str:
    patterns = [
        r"^\s*([\u4e00-\u9fff]{2,4})(?=\s*[-|｜—])",
        r"公开主页可见[:：]\s*([\u4e00-\u9fff]{2,4})",
        r"姓名[:：]\s*([\u4e00-\u9fff]{2,4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _clean(match.group(1))
    return ""


def _extract_after_label(text: str, labels: list[str]) -> str:
    for label in labels:
        match = re.search(rf"{label}\s*[:：]\s*([^,，。；;\n]+)", text)
        if match:
            return _clean(match.group(1))
    return ""


def _extract_school(text: str) -> str:
    labeled = _extract_after_label(text, ["学校", "学历学校", "毕业院校", "教育"])
    if labeled:
        return labeled
    match = re.search(r"([\u4e00-\u9fffA-Za-z]{2,30}(?:大学|学院|学校|University|College))", text)
    return _clean(match.group(1)) if match else ""


def _extract_title_company(text: str) -> tuple[str, str]:
    labeled_title = _extract_after_label(text, ["职位", "职务", "岗位", "title"])
    labeled_company = _extract_after_label(text, ["公司", "任职公司", "company"])
    if labeled_title or labeled_company:
        return labeled_title, labeled_company

    first_line = re.split(r"[。\n]", text, maxsplit=1)[0]
    parts = [_clean(part) for part in re.split(r"\s*[-|｜—]\s*", first_line) if _clean(part)]
    if len(parts) >= 3:
        return parts[1], parts[2].replace("脉脉", "").strip()

    company_match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,30}(?:公司|集团|科技|网络|信息|智能))", text)
    title_match = re.search(r"([\u4e00-\u9fffA-Za-z]{0,12}(?:产品经理|工程师|负责人|总监|经理|专家|顾问))", text)
    return (
        _clean(title_match.group(1)) if title_match else "",
        _clean(company_match.group(1)) if company_match else "",
    )


def extract_public_profile(result: dict[str, Any], optional_profile_text: str | None = None) -> dict[str, Any]:
    title = _clean(result.get("title"))
    snippet = _clean(result.get("snippet"))
    manual_text = _clean(optional_profile_text)
    raw_text = " ".join(part for part in [title, snippet, manual_text] if part)
    extracted_title, extracted_company = _extract_title_company(raw_text)
    raw_status = _clean(result.get("access_status") or "search_snippet_only")
    access_status = "manual_public_text" if manual_text else raw_status
    if raw_status in ACCESS_LIMITED_STATUSES and not manual_text:
        access_status = "access_limited"

    return {
        "name": _first_candidate_name(raw_text),
        "title": extracted_title,
        "company": extracted_company,
        "school": _extract_school(raw_text),
        "summary": snippet or manual_text,
        "profile_url": _canonical_url(_clean(result.get("url"))),
        "source_url": _clean(result.get("url")),
        "source_query": _clean(result.get("source_query")),
        "access_status": access_status,
        "raw_text": raw_text,
    }


def _contains(haystack: str, needle: Any) -> bool:
    clean = _normalize(needle)
    return bool(clean and clean in _normalize(haystack))


def _name_score(expected: str, profile: dict[str, Any]) -> int:
    name = _clean(profile.get("name"))
    raw = _clean(profile.get("raw_text"))
    if name and _normalize(name) == _normalize(expected):
        return 40
    if name and (_contains(name, expected) or _contains(expected, name)):
        return 15
    if _contains(raw, expected):
        return 40
    return 0


def _field_score(profile: dict[str, Any], field: str, expected: Any, full_score: int) -> int:
    expected_text = _clean(expected)
    if not expected_text:
        return 0
    haystack = " ".join(
        [
            _clean(profile.get(field)),
            _clean(profile.get("summary")),
            _clean(profile.get("raw_text")),
        ]
    )
    return full_score if _contains(haystack, expected_text) else 0


def _extra_keyword_score(profile: dict[str, Any], keywords: Any) -> int:
    haystack = " ".join([_clean(profile.get("summary")), _clean(profile.get("raw_text"))])
    return 5 if any(_contains(haystack, term) for term in _as_terms(keywords)) else 0


def _risk_note(scored: dict[str, Any]) -> str:
    missing = [
        label
        for key, label in [
            ("name", "姓名"),
            ("title", "职位"),
            ("school", "学校"),
            ("company", "公司"),
        ]
        if int(scored["score_breakdown"].get(key, 0)) <= 0
    ]
    notes: list[str] = []
    if scored.get("access_status") == "access_limited":
        notes.append("页面需要登录、验证码或授权访问，系统已停止自动读取，请 HR 在官方界面人工复核。")
    if missing:
        notes.append(f"{'、'.join(missing)}信息缺失或未命中，需人工确认。")
    if scored["confidence_score"] >= 80 and not missing:
        notes.append("高置信匹配，但仍需 HR 人工核对公开主页与简历信息。")
    return "；".join(notes) or "候选线索需 HR 人工复核后使用。"


def score_maimai_candidate(input_data: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    breakdown = {
        "name": _name_score(_clean(input_data.get("candidate_name")), profile),
        "title": _field_score(profile, "title", input_data.get("known_title"), SCORE_WEIGHTS["title"]),
        "school": _field_score(profile, "school", input_data.get("known_school"), SCORE_WEIGHTS["school"]),
        "company": _field_score(profile, "company", input_data.get("known_company"), SCORE_WEIGHTS["company"]),
        "extra_keywords": _extra_keyword_score(profile, input_data.get("extra_keywords")),
    }
    score = int(sum(breakdown.values()))
    matched_fields = [field for field, value in breakdown.items() if int(value) > 0]
    scored = dict(profile)
    scored.update(
        {
            "score_breakdown": breakdown,
            "confidence_score": score,
            "matched_fields": matched_fields,
            "manual_review_required": True,
        }
    )
    scored["risk_note"] = _risk_note(scored)
    return scored


def _manual_text_for_result(manual_profile_texts: Any, result: dict[str, Any]) -> str:
    if not manual_profile_texts:
        return ""
    url = _clean(result.get("url"))
    canonical = _canonical_url(url)
    if isinstance(manual_profile_texts, dict):
        return _clean(manual_profile_texts.get(url) or manual_profile_texts.get(canonical))
    if isinstance(manual_profile_texts, list):
        for item in manual_profile_texts:
            if isinstance(item, dict) and _canonical_url(_clean(item.get("url"))) == canonical:
                return _clean(item.get("text") or item.get("profile_text"))
    return _clean(manual_profile_texts)


def verify_maimai_candidate(
    input_data: dict[str, Any],
    search_results: list[dict[str, Any]],
    manual_profile_texts: Any | None = None,
) -> dict[str, Any]:
    queries = build_maimai_queries(input_data)
    normalized = normalize_search_results(search_results)
    candidates: list[dict[str, Any]] = []
    for result in normalized:
        manual_text = _manual_text_for_result(manual_profile_texts, result)
        profile = extract_public_profile(result, manual_text)
        candidates.append(score_maimai_candidate(input_data, profile))

    candidates.sort(key=lambda item: item["confidence_score"], reverse=True)
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank"] = index

    if len(candidates) > 1 and candidates[0]["confidence_score"] - candidates[1]["confidence_score"] < 10:
        candidates[0]["risk_note"] = f"{candidates[0]['risk_note']}；第一名与第二名分差较小，存在同名候选人风险。"

    best = candidates[0] if candidates and candidates[0]["confidence_score"] >= 40 else None
    return {
        "queries": queries,
        "best_match_url": best.get("profile_url") if best else None,
        "confidence_score": best.get("confidence_score", 0) if best else 0,
        "matched_fields": best.get("matched_fields", []) if best else [],
        "risk_note": best.get("risk_note") if best else "未找到足够可信的公开匹配结果，需人工检索确认。",
        "manual_review_required": True,
        "candidates": candidates,
        "compliance": {
            "public_data_only": True,
            "no_login_bypass": True,
            "no_captcha_bypass": True,
            "no_private_message": True,
            "manual_review_required": True,
        },
    }
