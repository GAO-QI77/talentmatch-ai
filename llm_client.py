from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    import requests
except Exception:  # pragma: no cover - allows local tests without installed deps
    requests = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from utils import calculate_mock_score, recommendation_from_score


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-chat"

if load_dotenv:
    load_dotenv()


def _session_api_key() -> str:
    try:
        import streamlit as st

        return st.session_state.get("DEEPSEEK_API_KEY", "")
    except Exception:
        return ""


def get_deepseek_api_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY", "").strip() or _session_api_key().strip()


def has_deepseek_api_key() -> bool:
    return bool(get_deepseek_api_key())


def _extract_candidate_id(prompt: str) -> str:
    match = re.search(r"\b(C[0-9A-Z]{3,}|M[0-9A-Z]{3,})\b", prompt, re.I)
    return match.group(1).upper() if match else ""


def _extract_name(prompt: str) -> str:
    for marker in ["candidate_name", "name", "姓名"]:
        match = re.search(marker + r"[:：\s]+([\u4e00-\u9fa5A-Za-z0-9_\- ]{2,20})", prompt)
        if match:
            return match.group(1).strip()
    chinese_names = re.findall(r"[\u4e00-\u9fa5]{2,4}", prompt)
    return chinese_names[0] if chinese_names else "候选人"


def infer_response_type(prompt: str, system_prompt: str = "") -> str:
    joined = f"{system_prompt}\n{prompt}".lower()
    if "message_draft" in joined or "触达" in joined or "私信" in joined:
        return "outreach"
    if "match_score" in joined or "匹配评分" in joined or "candidate_id" in joined:
        return "match"
    if "role_title" in joined or "jd" in joined or "岗位画像" in joined:
        return "jd"
    return "generic"


def mock_llm_response(
    user_prompt: str,
    response_type: str | None = None,
    error: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response_type = response_type or infer_response_type(user_prompt)
    context = context or {}

    if response_type == "jd":
        return {
            "role_title": context.get("role_title") or "AI-HR 产品实习生 / AI 招聘工具产品经理",
            "location": context.get("location") or "未明确",
            "seniority_level": "实习生 / 初级产品经理",
            "responsibilities": [
                "负责 AI 招聘工具的需求分析、功能设计和 Demo 搭建。",
                "使用大模型能力优化简历筛选、候选人匹配和招聘触达流程。",
                "设计候选人评分、推荐排序和招聘 CRM 流程。",
                "关注数据合规、隐私保护和用户体验。",
            ],
            "must_have_requirements": [
                "熟悉 AI 产品设计或 HR SaaS 产品。",
                "理解大模型 API 调用、Prompt 设计和结构化输出。",
                "具备基础 Python / Streamlit / 数据处理能力。",
                "具备良好沟通能力和产品文档能力。",
            ],
            "nice_to_have_requirements": [
                "有招聘、金融科技、SaaS 或数据分析项目经验。",
                "做过候选人评分、推荐排序、CRM 或招聘运营工具。",
            ],
            "excluded_conditions": ["不符合合规招聘要求或缺少公开/授权资料来源。"],
            "company_selling_points": [
                "AI 招聘工具从 0 到 1 的产品实践机会。",
                "能接触 HR、业务和研发多方协作场景。",
                "重视数据合规、隐私保护和可解释 AI。",
            ],
            "ideal_candidate_profile": "理解招聘场景，具备 AI 产品、数据处理和快速 Demo 搭建能力，能在合规前提下提升 HR 筛选和触达效率。",
            "scoring_weights": {
                "must_have": 35,
                "industry_background": 20,
                "skills_projects": 25,
                "career_path": 10,
                "outreach_priority": 10,
            },
            "_mock": True,
            "_fallback_error": error or "",
        }

    if response_type == "match":
        candidate = context.get("candidate") or {
            "candidate_id": _extract_candidate_id(user_prompt) or "C000",
            "name": _extract_name(user_prompt),
            "skills": user_prompt,
            "work_experience": user_prompt,
            "project_experience": user_prompt,
            "public_profile_text": user_prompt,
            "location": "未明确",
        }
        job_profile = context.get("job_profile") or {
            "must_have_requirements": ["AI 产品", "HR SaaS", "大模型 API", "Python"],
            "nice_to_have_requirements": ["招聘", "数据分析", "Streamlit"],
            "responsibilities": ["候选人匹配", "触达", "CRM"],
            "company_selling_points": ["AI 招聘工具", "合规"],
        }
        result = calculate_mock_score(job_profile, candidate)
        result["_mock"] = True
        result["_fallback_error"] = error or ""
        return result

    if response_type == "outreach":
        candidate = context.get("candidate") or {}
        job_profile = context.get("job_profile") or {}
        match = context.get("match_result") or {}
        style = context.get("message_style") or "专业正式"
        candidate_id = candidate.get("candidate_id") or _extract_candidate_id(user_prompt) or "C000"
        name = candidate.get("name") or _extract_name(user_prompt)
        role = job_profile.get("role_title") or "AI-HR 产品相关岗位"
        angle = match.get("outreach_angle") or "你过往经历与 AI 招聘工具方向有一定关联"
        prefix = {
            "专业正式": f"你好 {name}，我关注到{angle}。",
            "轻松自然": f"{name} 你好，看到你做过的方向和我们在看的 AI 招聘工具挺接近。",
            "猎头风格": f"{name} 你好，我这边有一个比较匹配你背景的 {role} 机会，想简单和你同步一下。",
        }.get(style, f"你好 {name}，我关注到{angle}。")
        return {
            "candidate_id": candidate_id,
            "message_style": style,
            "message_draft": (
                f"{prefix} 这个岗位主要围绕 JD 解析、候选人匹配和招聘 CRM，团队也很重视合规和人工确认。"
                "如果你最近愿意看看新方向，我可以发你更详细的信息；暂时不考虑也完全没关系。"
            ),
            "personalization_points": [
                angle,
                "结合岗位亮点表达机会，不夸大承诺。",
                "使用低压力 CTA，避免骚扰式触达。",
            ],
            "compliance_check": {
                "no_sensitive_inference": True,
                "not_spammy": True,
                "manual_confirmation_required": True,
            },
            "suggested_follow_up": "保存为待人工确认，由 HR 复核后再决定是否联系。",
            "_mock": True,
            "_fallback_error": error or "",
        }

    return {
        "summary": "Mock Demo 模式已返回结构化结果。",
        "_mock": True,
        "_fallback_error": error or "",
    }


def call_deepseek(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.2,
    response_type: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response_type = response_type or infer_response_type(prompt, system_prompt)
    api_key = get_deepseek_api_key()
    if not api_key or requests is None:
        return mock_llm_response(prompt, response_type=response_type, context=context)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek response JSON root is not an object.")
        parsed["_mock"] = False
        return parsed
    except Exception as exc:
        return mock_llm_response(
            prompt,
            response_type=response_type,
            error=str(exc),
            context=context,
        )
