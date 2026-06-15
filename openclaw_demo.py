from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from llm_client import call_deepseek, mock_llm_response
from sourcing import build_search_strategy, profile_completeness, run_mock_search
from utils import (
    calculate_mock_score,
    contactable_reason,
    ensure_candidate_defaults,
    next_best_action,
    normalize_text,
)


DEFAULT_SAMPLE_JD = """岗位名称：AI-HR 产品实习生 / AI 招聘工具产品经理
工作地点：上海 / 远程协作
行业方向：AI + HR SaaS

岗位职责：
1. 负责 AI 招聘工具的需求分析、功能设计和 Demo 搭建；
2. 使用大模型能力优化简历筛选、候选人匹配和招聘触达流程；
3. 与 HR、业务部门、研发协作，推动 AI-HR 场景落地；
4. 设计候选人评分、推荐排序和招聘 CRM 流程；
5. 关注数据合规、隐私保护和用户体验。

岗位要求：
1. 熟悉 AI 产品设计或 HR SaaS 产品；
2. 理解大模型 API 调用、Prompt 设计和结构化输出；
3. 具备基础 Python / Streamlit / 数据处理能力；
4. 有招聘、金融科技、SaaS 或数据分析项目经验优先；
5. 具备良好沟通能力和产品文档能力。
"""

JD_SYSTEM_PROMPT = """你是专业招聘产品分析助手。请把 JD 解析成结构化岗位画像，输出 JSON。"""

OUTREACH_SYSTEM_PROMPT = """你是专业招聘沟通助手。请生成自然、克制、个性化且必须人工确认的候选人触达草稿，输出 JSON。"""


def _default_candidate_path() -> Path:
    return Path(__file__).with_name("sample_candidates.csv")


def _read_text(path: str | None) -> str:
    if not path:
        return DEFAULT_SAMPLE_JD
    return Path(path).read_text(encoding="utf-8")


def load_candidates(path: str | None = None) -> list[dict[str, Any]]:
    candidate_path = Path(path) if path else _default_candidate_path()
    df = pd.read_csv(candidate_path, keep_default_na=False)
    candidates = [ensure_candidate_defaults(row) for row in df.to_dict(orient="records")]
    for candidate in candidates:
        candidate["profile_completeness"] = candidate.get("profile_completeness") or profile_completeness(candidate)
    return candidates


def _infer_job_context(job_text: str) -> dict[str, str]:
    role_title = "AI-HR 产品实习生 / AI 招聘工具产品经理"
    location = "上海 / 远程协作"
    for line in job_text.splitlines():
        clean = line.strip()
        if clean.startswith(("岗位名称", "职位名称")) and "：" in clean:
            role_title = clean.split("：", 1)[1].strip() or role_title
        if clean.startswith(("工作地点", "地点")) and "：" in clean:
            location = clean.split("：", 1)[1].strip() or location
    return {"role_title": role_title, "location": location}


def build_job_profile(job_text: str, mode: str) -> dict[str, Any]:
    context = _infer_job_context(job_text)
    if mode == "deepseek" and _deepseek_env_available():
        return call_deepseek(
            job_text,
            system_prompt=JD_SYSTEM_PROMPT,
            response_type="jd",
            context=context,
        )
    return mock_llm_response(job_text, response_type="jd", context=context)


def _build_outreach_draft(
    candidate: dict[str, Any],
    match_result: dict[str, Any],
    job_profile: dict[str, Any],
    mode: str,
) -> dict[str, Any] | None:
    contactable, _ = contactable_reason(candidate)
    if not contactable:
        return None
    context = {
        "candidate": candidate,
        "match_result": match_result,
        "job_profile": job_profile,
        "message_style": "专业正式",
    }
    if mode == "deepseek" and _deepseek_env_available():
        draft = call_deepseek(
            json.dumps(context, ensure_ascii=False),
            system_prompt=OUTREACH_SYSTEM_PROMPT,
            response_type="outreach",
            context=context,
        )
    else:
        draft = mock_llm_response("", response_type="outreach", context=context)
    draft["candidate_id"] = candidate.get("candidate_id")
    draft["candidate_name"] = candidate.get("name")
    draft["message_draft"] = _clean_text(str(draft.get("message_draft", "")))
    draft["confirmation_status"] = "待人工确认"
    draft["delivery_status"] = "未发送"
    return draft


def score_candidates(
    candidates: list[dict[str, Any]],
    job_profile: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        result = calculate_mock_score(job_profile, candidate)
        result["suggested_next_action"] = next_best_action(candidate, result)
        result["contactable"] = contactable_reason(candidate)[0]
        result["candidate"] = candidate
        scored.append(result)
    scored.sort(key=lambda item: int(item.get("match_score", 0) or 0), reverse=True)
    return scored[: max(1, top_k)]


def _candidate_pool_summary(candidates: list[dict[str, Any]], retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    guarded = [
        candidate
        for candidate in candidates
        if candidate.get("blacklist_status") == "是" or candidate.get("contact_status") in ["黑名单", "不再联系"]
    ]
    missing = [
        candidate
        for candidate in candidates
        if not normalize_text(candidate.get("current_title")) or not normalize_text(candidate.get("skills"))
    ]
    sources = sorted({str(candidate.get("source_type") or "unknown") for candidate in candidates})
    return {
        "total_candidates": len(candidates),
        "retrieved_leads": len(retrieved),
        "guarded_non_contactable": len(guarded),
        "profiles_with_missing_core_fields": len(missing),
        "source_types": sources,
    }


def build_demo_report(
    mode: str = "mock",
    job_file: str | None = None,
    candidate_file: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    deepseek_available = _deepseek_env_available()
    actual_mode = "deepseek" if mode == "deepseek" and deepseek_available else "mock"
    job_text = _read_text(job_file)
    candidates = load_candidates(candidate_file)
    job_profile = build_job_profile(job_text, actual_mode)
    strategy = build_search_strategy(_strategy_ready_profile(job_profile))
    retrieved = run_mock_search(strategy, candidates, limit=min(20, len(candidates)), source_type="local_talent_pool")
    top_results = score_candidates(candidates, job_profile, top_k)

    top_candidates: list[dict[str, Any]] = []
    outreach_drafts: list[dict[str, Any]] = []
    for result in top_results:
        candidate = result.pop("candidate")
        candidate_summary = {
            "candidate_id": candidate.get("candidate_id"),
            "name": candidate.get("name"),
            "current_company": candidate.get("current_company"),
            "current_title": candidate.get("current_title"),
            "location": candidate.get("location"),
            "match_score": result.get("match_score"),
            "recommendation_level": result.get("recommendation_level"),
            "confidence_level": result.get("confidence_level"),
            "evidence": result.get("evidence", []),
            "match_reasons": result.get("match_reasons", []),
            "risks": result.get("risks", []),
            "missing_information": result.get("missing_information", []),
            "suggested_next_action": result.get("suggested_next_action"),
            "contactable": result.get("contactable"),
            "contact_guardrail": contactable_reason(candidate)[1],
            "source_type": candidate.get("source_type"),
            "profile_completeness": candidate.get("profile_completeness") or profile_completeness(candidate),
        }
        top_candidates.append(candidate_summary)
        draft = _build_outreach_draft(candidate, result, job_profile, actual_mode)
        if draft:
            outreach_drafts.append(draft)

    return {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": actual_mode,
            "requested_mode": mode,
            "candidate_count": len(candidates),
            "top_k": top_k,
            "deepseek_available": deepseek_available,
        },
        "job_profile": job_profile,
        "search_strategy": strategy,
        "candidate_pool_summary": _candidate_pool_summary(candidates, retrieved),
        "top_candidates": top_candidates,
        "outreach_drafts": outreach_drafts,
        "compliance_guardrails": [
            "只使用本地 mock 数据、CSV 或用户主动提供的文本。",
            "不登录、不抓取、不绕过招聘平台权限或风控。",
            "不自动发送触达消息；所有草稿状态均为待人工确认。",
            "黑名单 / 不再联系候选人只参与分析，不生成触达草稿。",
            "不展示、打印、导出或持久化保存 API Key。",
        ],
        "interview_talk_track": [
            "这是一个 OpenClaw 可调用的 AI-HR Sourcing Copilot 技能演示。",
            "Agent 输入 JD 后，可以生成搜索策略、召回候选人、完成证据级匹配和触达草稿准备。",
            "我把自动化边界设计在草稿和 CRM 建议层，最终联系仍由 HR 人工确认。",
        ],
        "future_iteration_notes": [
            "V3：接入官方招聘平台 API、ATS、企业授权人才库和浏览器分享入口。",
            "V4：升级为 Sourcing Agent，支持任务状态、定期刷新、多源召回和失败重试。",
            "模型层：加入 embedding 召回、reranker 重排、逻辑回归 / LightGBM 排序和 HR 反馈闭环。",
            "工程层：迁移到 Next.js + FastAPI + PostgreSQL + pgvector，并增加权限、审计和队列。",
        ],
    }


def _deepseek_env_available() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def _clean_text(text: str) -> str:
    return text.replace("。。", "。").replace("。. ", "。")


def _strategy_ready_profile(job_profile: dict[str, Any]) -> dict[str, Any]:
    profile = dict(job_profile)
    excluded = [str(item) for item in profile.get("excluded_conditions", []) if str(item).strip()]
    if not excluded or any(len(item) > 12 or "合规" in item or "授权" in item for item in excluded):
        profile["excluded_conditions"] = ["纯销售", "无产品经验", "非招聘方向"]
    return profile


def _fmt_list(items: list[Any], limit: int = 4) -> str:
    clean = [str(item) for item in items if str(item).strip()]
    return "\n".join(f"- {item}" for item in clean[:limit]) if clean else "- 未明确"


def render_markdown(report: dict[str, Any]) -> str:
    profile = report["job_profile"]
    strategy = report["search_strategy"]
    summary = report["candidate_pool_summary"]
    lines: list[str] = [
        "# TalentMatch AI OpenClaw 面试展示报告",
        "",
        "## 一分钟产品介绍",
        "TalentMatch AI 是一个合规优先的 AI-HR Sourcing Copilot。OpenClaw Agent 可以从 JD 出发，自动生成搜索策略、召回候选人、完成证据级匹配评分，并输出待人工确认的触达草稿。",
        "",
        "## 本次 Demo 输入",
        f"- 岗位：{profile.get('role_title', '未明确')}",
        f"- 地点：{profile.get('location', '未明确')}",
        f"- 模式：{report['metadata']['mode']}",
        f"- 候选人池：{summary['total_candidates']} 人",
        f"- 不可触达护栏命中：{summary['guarded_non_contactable']} 人",
        "",
        "## 自动检索策略",
        "### 关键词",
        _fmt_list(strategy.get("keywords", []), 8),
        "",
        "### 布尔搜索式",
        _fmt_list(strategy.get("boolean_queries", []), 8),
        "",
        "## Top 候选人短名单",
    ]

    for index, candidate in enumerate(report["top_candidates"], start=1):
        evidence = candidate.get("evidence", [])
        first_evidence = evidence[0] if evidence else {}
        lines.extend(
            [
                f"### {index}. {candidate['name']}｜{candidate['current_company']}｜{candidate['current_title']}",
                f"- 匹配分：{candidate['match_score']} / 100",
                f"- 推荐等级：{candidate['recommendation_level']}；置信度：{candidate['confidence_level']}",
                f"- 下一步：{candidate['suggested_next_action']}",
                f"- 触达状态：{'可生成草稿' if candidate['contactable'] else '不可触达'}",
                f"- 关键证据：{first_evidence.get('job_requirement', '未明确')} → {first_evidence.get('candidate_evidence', '未明确')}",
                f"- 风险提示：{'；'.join(candidate.get('risks', [])[:2])}",
                "",
            ]
        )

    lines.extend(
        [
            "## 触达草稿示例",
        ]
    )
    for draft in report["outreach_drafts"][:3]:
        lines.extend(
            [
                f"### {draft.get('candidate_name')}｜{draft.get('confirmation_status')}",
                draft.get("message_draft", "暂无草稿"),
                "",
            ]
        )
    if not report["outreach_drafts"]:
        lines.extend(["暂无可触达候选人草稿。", ""])

    lines.extend(
        [
            "## 合规护栏",
            _fmt_list(report["compliance_guardrails"], 8),
            "",
            "## 未来迭代方向",
            _fmt_list(report["future_iteration_notes"], 8),
            "",
            "## 面试讲解要点",
            _fmt_list(report["interview_talk_track"], 8),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TalentMatch AI OpenClaw interview demo CLI")
    parser.add_argument("--mode", choices=["mock", "deepseek"], default="mock")
    parser.add_argument("--job-file", default=None)
    parser.add_argument("--candidate-file", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_demo_report(
        mode=args.mode,
        job_file=args.job_file,
        candidate_file=args.candidate_file,
        top_k=args.top_k,
    )
    rendered = render_markdown(report) if args.format == "markdown" else json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
