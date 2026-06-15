from __future__ import annotations

import json
from html import escape
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from llm_client import call_deepseek, has_deepseek_api_key
from maimai_verifier import (
    build_maimai_queries,
    parse_search_results_json,
    verify_maimai_candidate,
)
from sourcing import (
    build_search_strategy,
    candidate_from_sourcing_result,
    profile_completeness,
    run_mock_search,
)
from utils import (
    CRM_STATUSES,
    DAILY_OUTREACH_LIMIT,
    RECOMMENDATION_LEVELS,
    REQUIRED_CANDIDATE_FIELDS,
    SOURCE_TYPES,
    add_audit_log,
    build_match_evidence,
    build_compliance_status,
    build_candidate_from_profile_text,
    confidence_from_missing_info,
    contactable_reason,
    dataframe_for_export,
    dedupe_candidates,
    detect_sensitive_terms,
    ensure_candidate_defaults,
    candidate_identity_keys,
    next_best_action,
    now_text,
    stable_id,
)


SAMPLE_JD = """岗位名称：AI-HR 产品实习生 / AI 招聘工具产品经理
工作地点：上海 / 远程协作
薪资范围：面议
行业方向：AI + HR SaaS
公司阶段：早期产品验证阶段

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

公司卖点：
可以完整参与 AI 招聘工具从 0 到 1 的 Demo 设计，直接接触真实招聘效率提升场景，并在合规前提下验证 AI 产品价值。
"""

JD_SYSTEM_PROMPT = """你是专业招聘产品分析助手。你需要把非结构化 JD 解析成结构化岗位画像。你只能基于用户输入内容提取信息，不要编造。若信息缺失，请填入“未明确”。
输出 JSON 字段：
{
  "role_title": "",
  "location": "",
  "seniority_level": "",
  "responsibilities": [],
  "must_have_requirements": [],
  "nice_to_have_requirements": [],
  "excluded_conditions": [],
  "company_selling_points": [],
  "ideal_candidate_profile": "",
  "scoring_weights": {
    "must_have": 35,
    "industry_background": 20,
    "skills_projects": 25,
    "career_path": 10,
    "outreach_priority": 10
  }
}"""

MATCH_SYSTEM_PROMPT = """你是专业招聘分析助手。你只能根据用户提供的岗位信息和候选人公开/授权资料进行分析。不要编造候选人经历，不要推断敏感个人属性。请输出客观、可解释、可审计的匹配结果。若信息不足，请明确说明“不足以判断”。
输出 JSON 字段：
{
  "candidate_id": "",
  "candidate_name": "",
  "match_score": 0,
  "recommendation_level": "",
  "confidence_level": "",
  "evidence": [
    {
      "job_requirement": "",
      "candidate_evidence": ""
    }
  ],
  "score_breakdown": {
    "must_have": 0,
    "industry_background": 0,
    "skills_projects": 0,
    "career_path": 0,
    "outreach_priority": 0
  },
  "match_reasons": [],
  "risks": [],
  "missing_information": [],
  "interview_focus": [],
  "outreach_angle": "",
  "suggested_next_action": ""
}"""

OUTREACH_SYSTEM_PROMPT = """你是专业招聘沟通助手。你需要根据岗位信息、候选人背景和匹配理由，生成一条自然、克制、个性化的候选人触达私信草稿。不要夸张承诺，不要骚扰，不要涉及敏感个人属性，不要像群发。消息应该简短、礼貌、有明确岗位亮点，并给候选人低压力选择。
输出 JSON 字段：
{
  "candidate_id": "",
  "message_style": "",
  "message_draft": "",
  "personalization_points": [],
  "compliance_check": {
    "no_sensitive_inference": true,
    "not_spammy": true,
    "manual_confirmation_required": true
  },
  "suggested_follow_up": ""
}"""


def init_state() -> None:
    defaults: dict[str, Any] = {
        "jobs": [],
        "candidates": [],
        "match_results": {},
        "match_history": [],
        "outreach_drafts": {},
        "outreach_history": [],
        "sourcing_strategy": None,
        "sourcing_inbox": [],
        "sourcing_history": [],
        "maimai_verification_result": None,
        "audit_log": [],
        "job_profile_draft": None,
        "jd_text": SAMPLE_JD,
        "mock_candidates_seeded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def setup_page() -> None:
    st.set_page_config(
        page_title="TalentMatch AI",
        page_icon="TM",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --tm-bg: #f3f6f8;
            --tm-surface: #ffffff;
            --tm-soft: #f8fafc;
            --tm-ink: #111827;
            --tm-muted: #475467;
            --tm-subtle: #667085;
            --tm-line: #d0d7de;
            --tm-blue: #0066cc;
            --tm-mint: #087443;
            --tm-warn: #b54708;
            --tm-red: #b42318;
        }
        html,
        body,
        .stApp,
        div[data-testid="stAppViewContainer"],
        div[data-testid="stMain"],
        section[data-testid="stMain"] {
            background: linear-gradient(180deg, #fbfcfd 0%, var(--tm-bg) 64%, #eef4f2 100%) !important;
            color: var(--tm-ink) !important;
        }
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1240px; }
        h1, h2, h3, h4, h5, h6 { letter-spacing: 0; color: var(--tm-ink); }
        h1 { font-weight: 760; line-height: 1.06; }
        p, li, label, span, div[data-testid="stMarkdownContainer"] { color: var(--tm-ink); font-size: 15px; }
        div[data-testid="stMarkdownContainer"] *:not(code):not(pre),
        div[data-testid="stCaptionContainer"] *,
        div[data-testid="stWidgetLabel"] *,
        div[data-testid="stForm"] label,
        div[data-testid="stForm"] p {
            color: var(--tm-ink) !important;
        }
        header[data-testid="stHeader"] { background: transparent !important; }
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        #MainMenu {
            display: none !important;
            visibility: hidden !important;
        }
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        div[data-testid="stSidebar"] {
            background: #f8fafc !important;
            border-right: 1px solid var(--tm-line);
        }
        section[data-testid="stSidebar"] *,
        div[data-testid="stSidebar"] * {
            color: var(--tm-ink) !important;
        }
        section[data-testid="stSidebar"] input {
            color: #111827 !important;
            background: #fff !important;
            -webkit-text-fill-color: #111827 !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="radio"] {
            background: transparent !important;
        }
        section[data-testid="stSidebar"] [data-testid="stAlert"] {
            background: #fff !important;
            border: 1px solid #98a2b3 !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--tm-line);
            border-radius: 8px;
            padding: 15px 16px;
            background: #fff !important;
            box-shadow: 0 8px 24px rgba(16, 24, 40, .06);
        }
        div[data-testid="stMetric"] * { color: var(--tm-ink) !important; }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricLabel"] p {
            color: var(--tm-muted) !important;
            font-size: 14px !important;
            font-weight: 760 !important;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            color: var(--tm-ink) !important;
            font-weight: 820 !important;
        }
        .stButton button {
            border-radius: 8px;
            border: 1px solid #98a2b3 !important;
            min-height: 42px;
            font-weight: 730 !important;
            color: #111827 !important;
            background: #fff !important;
        }
        .stButton button:hover { border-color: var(--tm-blue); color: var(--tm-blue); background: #f5faff; }
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-secondary"] * {
            background: #fff !important;
            color: #111827 !important;
            border-color: #98a2b3 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButton-primary"] *,
        button[data-testid="stBaseButton-secondaryFormSubmit"],
        button[data-testid="stBaseButton-secondaryFormSubmit"] *,
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stFormSubmitButton"] button * {
            background: var(--tm-blue) !important;
            color: #ffffff !important;
            border-color: var(--tm-blue) !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .stTextInput input,
        .stTextArea textarea,
        .stDateInput input,
        div[data-testid="stDateInput"] input,
        div[data-baseweb="input"],
        div[data-baseweb="input"] input,
        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"] {
            border-color: #98a2b3 !important;
            color: #111827 !important;
            background: #fff !important;
            font-size: 15px !important;
        }
        div[data-baseweb="input"] *,
        div[data-baseweb="select"] *,
        div[data-testid="stDateInput"] *,
        div[data-testid="stSelectbox"] *,
        div[data-testid="stMultiSelect"] * {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        div[data-baseweb="input"],
        div[data-baseweb="select"],
        div[data-baseweb="select"] > div,
        div[data-testid="stDateInput"] div,
        div[data-testid="stSelectbox"] div,
        div[data-testid="stMultiSelect"] div {
            background-color: #fff !important;
        }
        div[data-baseweb="select"] svg,
        div[data-testid="stDateInput"] svg {
            color: #111827 !important;
            fill: #111827 !important;
        }
        .stTextInput button,
        div[data-testid="stTextInput"] button,
        div[data-testid="stTextInput"] button * {
            background: #111827 !important;
            color: #ffffff !important;
            border-color: #111827 !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .stSelectbox *,
        .stMultiSelect *,
        .stRadio *,
        .stCheckbox *,
        .stTextInput label,
        .stTextInput label *,
        .stTextArea label,
        .stTextArea label *,
        div[data-testid="stTextInput"] label,
        div[data-testid="stTextInput"] label *,
        div[data-testid="stTextArea"] label,
        div[data-testid="stTextArea"] label * {
            color: var(--tm-ink) !important;
        }
        .stRadio label,
        .stCheckbox label {
            min-height: 30px;
            align-items: center;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] *,
        div[data-baseweb="popover"] div,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] *,
        div[data-baseweb="calendar"],
        div[data-baseweb="calendar"] *,
        div[data-baseweb="select-dropdown"],
        div[data-baseweb="select-dropdown"] *,
        ul[role="listbox"],
        ul[role="listbox"] *,
        li[role="option"],
        li[role="option"] * {
            background: #fff !important;
            color: var(--tm-ink) !important;
            -webkit-text-fill-color: var(--tm-ink) !important;
        }
        li[role="option"][aria-selected="true"],
        li[role="option"]:hover,
        div[role="option"][aria-selected="true"],
        div[role="option"]:hover {
            background: #eff8ff !important;
            color: #111827 !important;
        }
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div[aria-selected],
        div[data-baseweb="select"] [class*="placeholder"],
        div[data-baseweb="select"] [class*="singleValue"],
        div[data-baseweb="select"] [class*="valueContainer"],
        div[data-baseweb="calendar"] button,
        div[data-baseweb="calendar"] button *,
        div[data-baseweb="calendar"] div[role="gridcell"],
        div[data-baseweb="calendar"] div[role="gridcell"] * {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        div[data-baseweb="calendar"] button {
            background: #fff !important;
            border-color: #d0d7de !important;
        }
        div[data-baseweb="calendar"] button[aria-selected="true"],
        div[data-baseweb="calendar"] button[aria-current="date"] {
            background: #0066cc !important;
            color: #fff !important;
            -webkit-text-fill-color: #fff !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid #cbd5e1 !important;
            background: #fff !important;
        }
        div[data-testid="stAlert"] *,
        div[data-testid="stAlert"] p { color: #111827 !important; font-size: 15px !important; }
        div[data-testid="stExpander"] {
            border: 1px solid var(--tm-line) !important;
            border-radius: 8px !important;
            background: #fff !important;
            overflow: hidden;
        }
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary *,
        div[data-testid="stExpander"] div,
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] li,
        div[data-testid="stExpander"] span {
            background-color: #fff !important;
            color: var(--tm-ink) !important;
        }
        div[data-testid="stExpander"] summary {
            min-height: 46px;
            border-bottom: 1px solid #eef2f6;
            font-weight: 760 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--tm-line);
            border-radius: 8px;
            overflow: hidden;
            background: #fff !important;
            min-height: 280px;
        }
        div[data-testid="stDataFrame"] > div,
        div[data-testid="stDataFrame"] iframe,
        div[data-testid="stDataFrame"] section,
        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stDataFrame"] [role="row"],
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"] {
            background: #fff !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        div[data-testid="stDataFrame"] *,
        div[data-testid="stTable"] *,
        div[data-testid="stDataFrameResizable"] *,
        div[data-testid="stDataFrameGlideDataEditor"] * {
            color: var(--tm-ink) !important;
            background-color: #fff !important;
            -webkit-text-fill-color: var(--tm-ink) !important;
        }
        div[data-testid="stDataFrame"] canvas,
        div[data-testid="stDataFrameGlideDataEditor"] canvas {
            background: #fff !important;
            color-scheme: light !important;
        }
        div[data-testid="stDataFrame"] div[class*="glide"],
        div[data-testid="stDataFrame"] div[class*="DataGrid"],
        div[data-testid="stDataFrame"] div[class*="row"],
        div[data-testid="stDataFrame"] div[class*="cell"],
        div[data-testid="stDataFrame"] div[class*="header"] {
            background-color: #fff !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] *,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] * {
            color: var(--tm-ink) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #fff !important;
            border: 1px dashed #98a2b3 !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button * {
            background: #fff !important;
            color: #111827 !important;
            border-color: #98a2b3 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        div[data-testid="stCodeBlock"],
        div[data-testid="stCodeBlock"] *,
        pre,
        pre *,
        code {
            background: #f8fafc !important;
            color: #111827 !important;
            border-color: #d0d7de !important;
            text-shadow: none !important;
        }
        button[data-testid="stBaseButton-primary"] div[data-testid="stMarkdownContainer"] *,
        button[data-testid="stBaseButton-secondaryFormSubmit"] div[data-testid="stMarkdownContainer"] *,
        div[data-testid="stFormSubmitButton"] button div[data-testid="stMarkdownContainer"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background: transparent !important;
        }
        .tm-hero {
            border: 1px solid var(--tm-line);
            border-radius: 8px;
            padding: 26px 28px;
            background:
                linear-gradient(120deg, #ffffff, #f6fbfa),
                radial-gradient(circle at top right, rgba(0,102,204,.10), transparent 30%);
            box-shadow: 0 18px 46px rgba(16, 24, 40, .09);
            margin: 6px 0 18px;
        }
        .tm-hero h1 { margin: 0 0 10px; font-size: 2.34rem; }
        .tm-hero p { margin: 0; color: var(--tm-muted); max-width: 800px; font-size: 1.03rem; line-height: 1.62; }
        .tm-card {
            border: 1px solid var(--tm-line);
            border-radius: 8px;
            padding: 18px;
            background: #fff;
            box-shadow: 0 12px 30px rgba(16, 24, 40, .075);
            margin-bottom: 14px;
        }
        .tm-card-title { font-size: 1.08rem; font-weight: 760; margin-bottom: 5px; color: var(--tm-ink); line-height: 1.34; }
        .tm-card-meta { color: var(--tm-muted); font-size: .94rem; line-height: 1.55; }
        .tm-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
        .tm-split { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }
        .tm-badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: .84rem;
            line-height: 1.3;
            color: #1f2937;
            background: #f8fafc;
            margin: 2px 4px 2px 0;
            white-space: nowrap;
            font-weight: 720;
        }
        .tm-badge.good { color: #067647; border-color: #75e0a7; background: #ecfdf3; }
        .tm-badge.info { color: #175cd3; border-color: #84caff; background: #eff8ff; }
        .tm-badge.warn { color: #93370d; border-color: #fdb022; background: #fffaeb; }
        .tm-badge.danger { color: #b42318; border-color: #f97066; background: #fef3f2; }
        .tm-score {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            color: var(--tm-ink);
            font-weight: 820;
            background: conic-gradient(var(--tm-blue) calc(var(--score) * 1%), #edf1f5 0);
            position: relative;
            flex: 0 0 auto;
        }
        .tm-score::after {
            content: "";
            position: absolute;
            inset: 6px;
            background: #fff;
            border-radius: 50%;
            box-shadow: inset 0 0 0 1px #eef1f4;
        }
        .tm-score span { position: relative; z-index: 1; font-size: 1.05rem; }
        .tm-step {
            border: 1px solid var(--tm-line);
            border-radius: 8px;
            background: #fff;
            padding: 15px;
            min-height: 112px;
            box-shadow: 0 8px 22px rgba(16, 24, 40, .045);
        }
        .tm-step-num { color: var(--tm-blue); font-weight: 780; font-size: .83rem; }
        .tm-step-title { font-weight: 760; margin: 4px 0; color: var(--tm-ink); }
        .tm-divider { height: 1px; background: var(--tm-line); margin: 14px 0; }
        .tm-small { color: var(--tm-muted); font-size: .92rem; line-height: 1.55; }
        .tm-trust {
            border-left: 3px solid var(--tm-mint);
            border-top: 1px solid #abefc6;
            border-right: 1px solid #abefc6;
            border-bottom: 1px solid #abefc6;
            background: #f0fdf8;
            padding: 13px 15px;
            border-radius: 8px;
            color: #064e3b;
            margin: 8px 0 16px;
        }
        .tm-copy {
            border: 1px dashed #c9d3df;
            background: #fbfcfd;
            border-radius: 8px;
            padding: 14px;
            color: #1d2733;
        }
        .tm-stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 14px 0;
        }
        .tm-stat {
            border: 1px solid var(--tm-line);
            background: var(--tm-soft);
            border-radius: 8px;
            padding: 11px 12px;
            min-height: 70px;
        }
        .tm-stat-label { color: var(--tm-muted); font-size: .84rem; font-weight: 720; margin-bottom: 4px; }
        .tm-stat-value { color: var(--tm-ink); font-size: 1.05rem; font-weight: 820; line-height: 1.25; }
        .tm-two-col {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }
        .tm-section-label {
            color: var(--tm-ink);
            font-size: .9rem;
            font-weight: 780;
            margin-bottom: 5px;
        }
        .tm-list {
            margin: 0;
            padding-left: 18px;
            color: var(--tm-ink);
        }
        .tm-list li { color: var(--tm-ink); line-height: 1.55; margin: 2px 0; }
        .tm-query {
            margin-top: 12px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: #fbfdff;
            padding: 10px 12px;
        }
        .tm-query summary { color: #175cd3; cursor: pointer; font-weight: 760; font-size: .92rem; }
        .tm-query div { color: var(--tm-muted); font-size: .9rem; line-height: 1.55; margin-top: 8px; word-break: break-word; }
        .tm-cta {
            border: 1px solid #84caff;
            background: linear-gradient(135deg, #ffffff, #eef7ff);
            box-shadow: 0 14px 34px rgba(0, 102, 204, .12);
        }
        .tm-source-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 18px;
        }
        .tm-source-item {
            border: 1px solid var(--tm-line);
            border-radius: 8px;
            background: #fff;
            padding: 12px 13px;
        }
        .tm-source-name { color: var(--tm-ink); font-weight: 760; margin-bottom: 4px; }
        .tm-source-desc { color: var(--tm-muted); font-size: .9rem; line-height: 1.5; }
        .tm-query-card {
            border: 1px solid #dbe7f3;
            background: #f8fbff;
            border-radius: 8px;
            padding: 12px 13px;
            margin-bottom: 8px;
            color: #111827;
            font-size: .94rem;
            line-height: 1.55;
            word-break: break-word;
        }
        .tm-table-wrap {
            border: 1px solid var(--tm-line);
            border-radius: 8px;
            background: #fff;
            overflow: auto;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.5);
        }
        .tm-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            color: #111827;
            font-size: 14px;
            line-height: 1.45;
            min-width: 920px;
        }
        .tm-table th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #f8fafc;
            color: #344054;
            font-weight: 780;
            text-align: left;
            border-bottom: 1px solid #d0d7de;
            padding: 10px 12px;
            white-space: nowrap;
        }
        .tm-table td {
            background: #fff;
            color: #111827;
            border-bottom: 1px solid #eef2f6;
            padding: 10px 12px;
            vertical-align: top;
            max-width: 280px;
            word-break: break-word;
        }
        .tm-table tr:hover td { background: #f8fbff; }
        .tm-table-empty {
            padding: 18px;
            color: #475467;
            background: #fff;
            border: 1px solid var(--tm-line);
            border-radius: 8px;
        }
        @media (max-width: 900px) {
            .tm-stat-grid, .tm-two-col, .tm-source-strip { grid-template-columns: 1fr; }
            .tm-split { flex-direction: column; }
            .tm-score { width: 64px; height: 64px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> str:
    st.sidebar.title("TalentMatch AI")
    st.sidebar.caption("合规 AI 招聘 Copilot 工作台")
    api_key = st.sidebar.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="可选；不填则使用 Mock Demo 模式",
        key="api_key_input",
    )
    if api_key:
        st.session_state["DEEPSEEK_API_KEY"] = api_key.strip()

    mode = "DeepSeek API 模式" if has_deepseek_api_key() else "Mock Demo 模式"
    if mode == "DeepSeek API 模式":
        st.sidebar.success(mode)
    else:
        st.sidebar.warning(mode)
        st.sidebar.info("请提供 DeepSeek API Key 后继续运行真实 AI 匹配功能；你也可以继续使用 mock 模式演示。")

    page = st.sidebar.radio(
        "工作流",
        [
            "今日驾驶舱",
            "0 自动检索",
            "脉脉候选人验证",
            "1 定义岗位",
            "2 导入候选人",
            "3 评估候选人",
            "候选人详情",
            "确认触达",
            "来源与授权中心",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption("脉脉合规辅助工作流：只接收合规资料、生成草稿，不抓取、不自动发送。")
    return page


def render_header(title: str, description: str) -> None:
    mode = "DeepSeek API 模式" if has_deepseek_api_key() else "Mock Demo 模式"
    mode_badge = "DeepSeek API 模式" if mode == "DeepSeek API 模式" else "Mock Demo 模式 · 演示评分"
    st.markdown(
        f"""
        <div class="tm-hero">
          <div class="tm-row">
            <span class="tm-badge {'good' if mode == 'DeepSeek API 模式' else 'warn'}">{escape(mode_badge)}</span>
            <span class="tm-badge info">草稿生成 · HR 人工确认</span>
            <span class="tm-badge good">合规导入，不自动发送</span>
          </div>
          <h1>{escape(title)}</h1>
          <p>{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if mode != "DeepSeek API 模式":
        st.warning("请提供 DeepSeek API Key 后继续运行真实 AI 匹配功能；你也可以继续使用 mock 模式演示。")


def html_list(items: list[Any], limit: int = 3) -> str:
    clean = [escape(str(item)) for item in items if str(item).strip()]
    if not clean:
        return '<span class="tm-small">暂无明确证据</span>'
    return "".join(f'<li>{item}</li>' for item in clean[:limit])


def badge(label: str, tone: str = "") -> str:
    return f'<span class="tm-badge {tone}">{escape(label)}</span>'


def stat_box(label: str, value: Any) -> str:
    return (
        '<div class="tm-stat">'
        f'<div class="tm-stat-label">{escape(label)}</div>'
        f'<div class="tm-stat-value">{escape(str(value))}</div>'
        "</div>"
    )


def render_workflow_strip(active: int = 0) -> None:
    steps = [
        ("01", "定义岗位", "输入 JD，生成结构化岗位画像和评分权重。"),
        ("02", "评估候选人", "导入合规资料，查看证据级匹配解释。"),
        ("03", "确认触达", "生成低压力草稿，由 HR 人工确认后复制到合规渠道。"),
    ]
    cols = st.columns(3)
    for index, (num, title, desc) in enumerate(steps):
        tone = "info" if index == active else ""
        cols[index].markdown(
            f"""
            <div class="tm-step">
              <div class="tm-step-num">{num}</div>
              <div class="tm-step-title">{escape(title)}</div>
              <div class="tm-card-meta">{escape(desc)}</div>
              <div style="margin-top:10px">{badge('当前步骤' if index == active else '工作流', tone)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_trust_panel(candidate: dict[str, Any] | None = None, outreach_used: int = 0) -> None:
    if candidate:
        status = build_compliance_status(candidate, outreach_used, DAILY_OUTREACH_LIMIT)
        sensitive = status["sensitive_findings"] or ["未发现敏感属性"]
        contact_tone = "good" if status["contactable"] else "danger"
        contact_text = "可生成草稿" if status["contactable"] else "禁止触达"
        st.markdown(
            f"""
            <div class="tm-trust">
              <div class="tm-row">
                {badge('来源：' + str(status['data_source']), 'info')}
                {badge('敏感检查：' + '、'.join(sensitive), 'warn' if status['sensitive_findings'] else 'good')}
                {badge('今日额度剩余：' + str(status['remaining_quota']), 'info')}
                {badge(contact_text, contact_tone)}
                {badge('人工确认必需', 'good')}
              </div>
              <div class="tm-small" style="margin-top:6px">{escape(status['contact_guardrail'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="tm-trust">
              <div class="tm-row">
                <span class="tm-badge good">合规导入</span>
                <span class="tm-badge info">草稿复制到官方 App / 授权渠道</span>
                <span class="tm-badge warn">不抓取、不群发、不绕过平台规则</span>
                <span class="tm-badge good">人工确认必需</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_candidate_card(candidate: dict[str, Any], match: dict[str, Any] | None = None) -> None:
    match = match or {}
    score = int(match.get("match_score", 0) or 0)
    recommendation = match.get("recommendation_level", "未评分")
    confidence = match.get("confidence_level", "待评估")
    action = next_best_action(candidate, match)
    score_markup = (
        f'<div class="tm-score" style="--score:{score}"><span>{score if score else "-"}</span></div>'
    )
    source = candidate.get("source_type", "manual_input")
    status = candidate.get("contact_status", "未触达")
    retrieval_source = candidate.get("retrieval_source", source)
    retrieval_query = candidate.get("retrieval_query", "")
    completeness = candidate.get("profile_completeness") or profile_completeness(candidate)
    source_confidence = candidate.get("source_confidence", "中")
    reasons = match.get("match_reasons", []) or [candidate.get("source_note", "等待 AI 评分后展示证据级匹配理由")]
    risks = match.get("risks", [])[:2]
    mode = "演示评分" if match.get("_mock") else "DeepSeek 评分" if match else "未评分"
    contactable, guardrail = contactable_reason(candidate)
    guardrail_tone = "good" if contactable else "danger"
    source_label = f"{retrieval_source} · {source_confidence}"
    score_value = f"{score} / 100" if score else "待评分"
    query_text = retrieval_query or "非自动检索导入"
    st.markdown(
        f"""
        <div class="tm-card">
          <div class="tm-split">
            <div>
              <div class="tm-card-title">{escape(str(candidate.get('name', '未命名候选人')))}</div>
              <div class="tm-card-meta">
                {escape(str(candidate.get('current_company', '未明确')))} ·
                {escape(str(candidate.get('current_title', '未明确')))} ·
                {escape(str(candidate.get('location', '未明确')))}
              </div>
              <div class="tm-row" style="margin-top:10px">
                {badge(str(recommendation), 'info' if score >= 70 else 'warn')}
                {badge('下一步：' + action, 'good' if action in ['生成草稿', '人工确认', '安排面试'] else 'warn')}
                {badge('可联系' if contactable else '不可联系', guardrail_tone)}
              </div>
            </div>
            {score_markup}
          </div>
          <div class="tm-stat-grid">
            {stat_box('匹配分', score_value)}
            {stat_box('资料完整度', str(completeness) + ' / 100')}
            {stat_box('来源可信度', source_label)}
          </div>
          <div class="tm-row">
            {badge('置信度：' + str(confidence), 'good' if confidence == '高' else 'warn')}
            {badge(str(mode), 'warn' if mode == '演示评分' else 'good')}
            {badge('CRM：' + str(status), '')}
          </div>
          <div class="tm-divider"></div>
          <div class="tm-two-col">
            <div>
              <div class="tm-section-label">命中原因</div>
              <ul class="tm-list">{html_list(reasons, 2)}</ul>
            </div>
            <div>
              <div class="tm-section-label">风险 / 下一步</div>
              <ul class="tm-list">{html_list(risks or [guardrail], 2)}</ul>
            </div>
          </div>
          <details class="tm-query">
            <summary>查看命中搜索式与来源</summary>
            <div>{escape(str(query_text))}</div>
            <div>来源类型：{escape(str(source))}；检索时间：{escape(str(candidate.get('retrieved_at', '未记录')))}</div>
          </details>
        </div>
        """,
        unsafe_allow_html=True,
    )


def candidate_label(candidate: dict[str, Any]) -> str:
    return f"{candidate.get('candidate_id')} | {candidate.get('name')} | {candidate.get('current_title')}"


def job_label(job: dict[str, Any]) -> str:
    profile = job.get("profile", {})
    return f"{job.get('job_id')} | {profile.get('role_title', '未命名岗位')}"


def find_candidate(candidate_id: str) -> dict[str, Any] | None:
    for candidate in st.session_state.candidates:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    return None


def find_job(job_id: str) -> dict[str, Any] | None:
    for job in st.session_state.jobs:
        if job.get("job_id") == job_id:
            return job
    return None


def update_candidate(candidate_id: str, updates: dict[str, Any]) -> None:
    for index, candidate in enumerate(st.session_state.candidates):
        if candidate.get("candidate_id") == candidate_id:
            updated = dict(candidate)
            updated.update(updates)
            updated["last_updated"] = now_text()
            st.session_state.candidates[index] = ensure_candidate_defaults(updated)
            return


def add_candidates(new_candidates: list[dict[str, Any]], action: str) -> list[str]:
    normalized = [ensure_candidate_defaults(item) for item in new_candidates]
    before = len(st.session_state.candidates)
    combined = st.session_state.candidates + normalized
    unique, warnings = dedupe_candidates(combined)
    st.session_state.candidates = unique
    added = max(0, len(unique) - before)
    add_audit_log(st.session_state.audit_log, action, f"尝试导入 {len(normalized)} 人，新增 {added} 人，重复 {len(warnings)} 条。")
    return warnings


def dataframe_from_candidates(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        match = st.session_state.match_results.get(candidate.get("candidate_id"), {})
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "姓名": candidate.get("name"),
                "公司": candidate.get("current_company"),
                "职位": candidate.get("current_title"),
                "地点": candidate.get("location"),
                "年限": candidate.get("years_of_experience"),
                "匹配分": match.get("match_score", ""),
                "推荐等级": match.get("recommendation_level", ""),
                "关键匹配理由": "；".join(match.get("match_reasons", [])[:2]),
                "风险提示": "；".join(match.get("risks", [])[:2]),
                "触达状态": candidate.get("contact_status"),
                "黑名单": candidate.get("blacklist_status"),
                "来源": candidate.get("source_type"),
            }
        )
    return pd.DataFrame(rows)


def render_readable_table(df: pd.DataFrame, height: int = 420) -> None:
    if df.empty:
        st.markdown('<div class="tm-table-empty">暂无可展示数据。</div>', unsafe_allow_html=True)
        return

    display_df = df.fillna("").copy()
    for column in display_df.columns:
        display_df[column] = display_df[column].map(
            lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        )

    header = "".join(f"<th>{escape(str(column))}</th>" for column in display_df.columns)
    body_rows = []
    for _, row in display_df.iterrows():
        cells = "".join(f"<td>{escape(str(row[column]))}</td>" for column in display_df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        f"""
        <div class="tm-table-wrap" style="max-height:{height}px">
          <table class="tm-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_match_result(
    result: dict[str, Any],
    job_profile: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(result)
    normalized["candidate_id"] = candidate.get("candidate_id")
    normalized["candidate_name"] = candidate.get("name")
    normalized.setdefault("match_reasons", [])
    normalized.setdefault("risks", [])
    normalized.setdefault("missing_information", [])
    if not normalized.get("evidence"):
        normalized["evidence"] = build_match_evidence(job_profile, candidate)
    if not normalized.get("confidence_level"):
        normalized["confidence_level"] = confidence_from_missing_info(
            normalized.get("missing_information", []),
            normalized.get("evidence", []),
        )
    normalized.setdefault("suggested_next_action", next_best_action(candidate, normalized))
    return normalized


def active_job_profile() -> dict[str, Any]:
    if st.session_state.jobs:
        return st.session_state.jobs[-1].get("profile", {})
    if st.session_state.job_profile_draft:
        return st.session_state.job_profile_draft
    return {
        "role_title": "AI-HR 产品实习生 / AI 招聘工具产品经理",
        "location": "上海 / 北京 / 远程",
        "must_have_requirements": ["AI 产品设计", "HR SaaS", "大模型 API", "Python", "Streamlit"],
        "nice_to_have_requirements": ["招聘运营", "数据分析", "B2B SaaS"],
        "responsibilities": ["JD 解析", "候选人匹配", "触达草稿", "招聘 CRM"],
        "excluded_conditions": ["纯销售", "无产品经验"],
        "company_selling_points": ["AI 招聘工具", "合规", "从 0 到 1"],
    }


def merge_new_sourcing_leads(leads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = st.session_state.candidates + st.session_state.sourcing_inbox
    existing_keys = set()
    for candidate in existing:
        for key in candidate_identity_keys(candidate):
            existing_keys.add(key)

    fresh: list[dict[str, Any]] = []
    seen = set(existing_keys)
    for lead in leads:
        keys = candidate_identity_keys(lead)
        if any(key in seen for key in keys):
            continue
        fresh.append(lead)
        for key in keys:
            seen.add(key)
    return fresh


def sourcing_pipeline_summary() -> dict[str, int]:
    inbox = st.session_state.sourcing_inbox
    discovered = len(inbox) + len([c for c in st.session_state.candidates if c.get("retrieval_source") in ["mock_search", "local_talent_pool", "search_api", "ats"]])
    pending = len([lead for lead in inbox if lead.get("sourcing_status") in ["新发现", "待复核"]])
    added = len([c for c in st.session_state.candidates if c.get("sourcing_status") in ["已加入短名单", "已入库"] and c.get("retrieval_source") in ["mock_search", "local_talent_pool", "search_api", "ats"]])
    return {"discovered": discovered, "pending": pending, "added": added}


def render_search_strategy(strategy: dict[str, Any]) -> None:
    cols = st.columns(3)
    cols[0].markdown(
        f"""
        <div class="tm-card">
          <div class="tm-card-title">关键词</div>
          <div class="tm-card-meta">用于召回技能、行业和项目经历。</div>
          <div class="tm-divider"></div>
          <div>{''.join(badge(item, 'info') for item in strategy.get('keywords', [])[:10])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        f"""
        <div class="tm-card">
          <div class="tm-card-title">职位 / 地区</div>
          <div class="tm-card-meta">把岗位名扩展成招聘搜索里的常见表达。</div>
          <div class="tm-divider"></div>
          <div>{''.join(badge(item, '') for item in strategy.get('title_aliases', [])[:5])}</div>
          <div style="margin-top:8px">{''.join(badge(item, 'good') for item in strategy.get('locations', [])[:4])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        f"""
        <div class="tm-card">
          <div class="tm-card-title">目标行业 / 排除词</div>
          <div class="tm-card-meta">提升召回质量，减少明显不相关线索。</div>
          <div class="tm-divider"></div>
          <div>{''.join(badge(item, 'info') for item in strategy.get('industries', [])[:5])}</div>
          <div style="margin-top:8px">{''.join(badge(item, 'warn') for item in strategy.get('excluded_keywords', [])[:4])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("生成的布尔搜索式", expanded=True):
        for query in strategy.get("boolean_queries", []):
            st.markdown(f'<div class="tm-query-card">{escape(str(query))}</div>', unsafe_allow_html=True)


def page_auto_sourcing() -> None:
    render_header("自动检索", "从 JD 自动生成搜索策略，模拟多源召回候选人线索，并送入 Sourcing Inbox 由 HR 复核。")
    render_workflow_strip(active=1)
    st.markdown(
        """
        <div class="tm-source-strip">
          <div class="tm-source-item">
            <div class="tm-source-name">Mock Search</div>
            <div class="tm-source-desc">已启用，用于稳定演示自动发现候选人。</div>
          </div>
          <div class="tm-source-item">
            <div class="tm-source-name">本地人才库</div>
            <div class="tm-source-desc">已启用，从当前候选人池中做二次召回。</div>
          </div>
          <div class="tm-source-item">
            <div class="tm-source-name">官方 API Connector</div>
            <div class="tm-source-desc">预留入口，只接授权来源，不做非官方抓取。</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile = active_job_profile()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前岗位", profile.get("role_title", "示例岗位"))
    col2.metric("候选人池", len(st.session_state.candidates))
    col3.metric("Inbox 线索", len(st.session_state.sourcing_inbox))
    col4.metric("检索式", len((st.session_state.sourcing_strategy or build_search_strategy(profile)).get("boolean_queries", [])))

    if st.button("生成搜索策略"):
        st.session_state.sourcing_strategy = build_search_strategy(profile)
        add_audit_log(st.session_state.audit_log, "生成搜索策略", f"岗位：{profile.get('role_title', '示例岗位')}")
        st.success("已生成自动检索策略。")

    strategy = st.session_state.sourcing_strategy or build_search_strategy(profile)
    st.session_state.sourcing_strategy = strategy
    render_search_strategy(strategy)

    col1, col2 = st.columns([1, 1])
    if col1.button("运行 Mock 自动检索", type="primary"):
        sample_pool = load_sample_candidates()
        leads = run_mock_search(strategy, sample_pool, limit=8, source_type="mock_search")
        local_leads = run_mock_search(strategy, st.session_state.candidates, limit=5, source_type="local_talent_pool")
        fresh = merge_new_sourcing_leads(leads + local_leads)
        st.session_state.sourcing_inbox = fresh + st.session_state.sourcing_inbox
        st.session_state.sourcing_history.insert(
            0,
            {
                "time": now_text(),
                "source": "mock_search + local_talent_pool",
                "query_count": len(strategy.get("boolean_queries", [])),
                "new_leads": len(fresh),
            },
        )
        add_audit_log(st.session_state.audit_log, "自动检索", f"新增 {len(fresh)} 条候选人线索。")
        st.success(f"自动检索完成，新增 {len(fresh)} 条线索。")

    if col2.button("清空已忽略线索"):
        st.session_state.sourcing_inbox = [
            lead for lead in st.session_state.sourcing_inbox if lead.get("sourcing_status") != "已忽略"
        ]
        st.info("已清理已忽略线索。")

    st.subheader("Sourcing Inbox")
    st.caption("每条线索先进入复核队列，只显示关键摘要和两个动作：加入短名单或忽略。")
    if not st.session_state.sourcing_inbox:
        st.markdown(
            """
            <div class="tm-card tm-cta">
              <div class="tm-card-title">暂无新线索</div>
              <div class="tm-card-meta">点击“运行 Mock 自动检索”，系统会基于 JD 策略从 mock source 和本地人才库召回候选人。</div>
              <div style="margin-top:10px"><span class="tm-badge info">新发现</span><span class="tm-badge good">人工复核入库</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for index, lead in enumerate(st.session_state.sourcing_inbox):
        render_candidate_card(lead, st.session_state.match_results.get(lead.get("candidate_id")))
        action_cols = st.columns([1, 1, 4])
        if action_cols[0].button("加入短名单", key=f"add_sourcing_{lead.get('candidate_id')}_{index}"):
            candidate = candidate_from_sourcing_result(lead)
            candidate["sourcing_status"] = "已加入短名单"
            warnings = add_candidates([candidate], "自动检索加入短名单")
            st.session_state.sourcing_inbox[index]["sourcing_status"] = "已加入短名单"
            for warning in warnings:
                st.warning(warning)
            st.success(f"已加入候选人池：{candidate.get('name')}")
            st.rerun()
        if action_cols[1].button("忽略", key=f"ignore_sourcing_{lead.get('candidate_id')}_{index}"):
            st.session_state.sourcing_inbox[index]["sourcing_status"] = "已忽略"
            add_audit_log(st.session_state.audit_log, "忽略自动检索线索", f"{lead.get('candidate_id')} 已忽略。")
            st.rerun()


def page_dashboard() -> None:
    render_header("今日招聘驾驶舱", "从岗位定义、候选人评估到人工确认触达，用一个克制可信的工作流完成招聘判断。")
    render_workflow_strip(active=0 if not st.session_state.jobs else 1 if not st.session_state.match_results else 2)
    render_trust_panel()
    candidates = st.session_state.candidates
    matches = list(st.session_state.match_results.values())
    high_matches = [m for m in matches if int(m.get("match_score", 0) or 0) >= 85]
    pending = [c for c in candidates if c.get("contact_status") in ["未触达", "已生成话术", "待人工确认"]]
    contacted = [c for c in candidates if c.get("contact_status") in ["已联系", "已回复", "感兴趣", "进入面试"]]
    interested = [c for c in candidates if c.get("contact_status") == "感兴趣"]
    used_quota = len(st.session_state.outreach_history)
    remaining = max(0, DAILY_OUTREACH_LIMIT - used_quota)
    sourcing_summary = sourcing_pipeline_summary()
    conversion = "0%"
    if sourcing_summary["discovered"]:
        conversion = f"{int(sourcing_summary['added'] / sourcing_summary['discovered'] * 100)}%"

    st.markdown(
        f"""
        <div class="tm-card tm-cta">
          <div class="tm-split">
            <div>
              <div class="tm-card-title">从 JD 自动检索候选人</div>
              <div class="tm-card-meta">已自动载入 {len(candidates)} 条 mock 候选人。生成搜索策略，召回候选人线索，再由 HR 复核加入短名单。当前待复核 {sourcing_summary["pending"]} 条。</div>
            </div>
            <div>{badge('进入 0 自动检索', 'info')}{badge('人工复核', 'good')}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("自动发现", sourcing_summary["discovered"])
    metric_cols[1].metric("待复核", sourcing_summary["pending"])
    metric_cols[2].metric("高匹配", len(high_matches))
    metric_cols[3].metric("待确认", len([c for c in candidates if c.get("contact_status") == "待人工确认"]))

    left, middle, right = st.columns([1.05, 1.5, 1.05])
    with left:
        st.subheader("Sourcing Pipeline")
        st.markdown(
            f"""
            <div class="tm-card">
              <div class="tm-stat-grid">
                {stat_box('线索转化率', conversion)}
                {stat_box('候选人池', len(candidates))}
                {stat_box('剩余额度', remaining)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("岗位进度")
        if st.session_state.jobs:
            for job in st.session_state.jobs[-3:]:
                profile = job.get("profile", {})
                st.markdown(
                    f"""
                    <div class="tm-card">
                      <div class="tm-card-title">{escape(str(profile.get('role_title', '未命名岗位')))}</div>
                      <div class="tm-card-meta">{escape(str(profile.get('location', '未明确')))} · {escape(str(profile.get('seniority_level', '未明确')))}</div>
                      <div style="margin-top:10px">{badge('已生成岗位画像', 'good')}{badge('评分权重已就绪', 'info')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="tm-card">
                  <div class="tm-card-title">下一步：定义岗位</div>
                  <div class="tm-card-meta">点击侧边栏“1 定义岗位”，加载示例 JD 并生成结构化画像。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with middle:
        st.subheader("高匹配候选人")
        top_ids = [m.get("candidate_id") for m in sorted(matches, key=lambda item: int(item.get("match_score", 0) or 0), reverse=True)[:4]]
        top_candidates = [candidate for candidate in candidates if candidate.get("candidate_id") in top_ids]
        if top_candidates:
            for candidate in top_candidates:
                render_candidate_card(candidate, st.session_state.match_results.get(candidate.get("candidate_id")))
        else:
            st.markdown(
                """
                <div class="tm-card">
                  <div class="tm-card-title">还没有评分结果</div>
                  <div class="tm-card-meta">导入候选人后运行 AI 匹配，这里会展示最值得人工复核的候选人。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.subheader("确认与合规")
        pending_drafts = [c for c in candidates if c.get("contact_status") == "待人工确认"]
        st.markdown(
            f"""
            <div class="tm-card">
              <div class="tm-card-title">今日待办</div>
              <div class="tm-stat-grid">
                {stat_box('待确认草稿', len(pending_drafts))}
                {stat_box('已联系', len(contacted))}
                {stat_box('感兴趣', len(interested))}
              </div>
            </div>
            <div class="tm-card">
              <div class="tm-card-title">触达护栏</div>
              <div class="tm-card-meta">有 {len(pending_drafts)} 条草稿等待 HR 复核。不会自动发送。</div>
              <div style="margin-top:10px">{badge('草稿复制到官方 App / 授权渠道', 'info')}{badge('不群发', 'good')}</div>
            </div>
            <div class="tm-card">
              <div class="tm-card-title">Mock 回复率</div>
              <div class="tm-card-meta">18% · 仅用于现场演示，不代表真实触达表现。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.sourcing_history:
            latest = st.session_state.sourcing_history[0]
            st.markdown(
                f"""
                <div class="tm-card">
                  <div class="tm-card-title">最近自动检索</div>
                  <div class="tm-card-meta">{escape(str(latest.get('time')))} · 新增 {escape(str(latest.get('new_leads')))} 条线索</div>
                  <div style="margin-top:10px">{badge('Sourcing Copilot', 'info')}{badge('待 HR 复核', 'warn')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("最近审计日志", expanded=False):
        if st.session_state.audit_log:
            st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True, hide_index=True, height=260)
        else:
            st.caption("暂无审计日志。")


def read_uploaded_text(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""
    data = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".csv"):
        return data.decode("utf-8-sig", errors="ignore")
    return data.decode("utf-8", errors="ignore")


def page_job_creation() -> None:
    render_header("定义岗位", "先让系统理解岗位，再进入候选人评估；这是整个 Copilot 工作流的第一步。")
    render_workflow_strip(active=0)
    if st.button("填充示例 JD"):
        st.session_state.jd_text = SAMPLE_JD
        st.success("已填充示例 JD。")

    uploaded = st.file_uploader("上传 JD 文档（Demo 支持 txt / md / csv）", type=["txt", "md", "csv"])
    if uploaded:
        st.session_state.jd_text = read_uploaded_text(uploaded)
        st.info(f"已读取上传文件：{uploaded.name}")

    with st.form("job_form"):
        col1, col2, col3 = st.columns(3)
        role_title = col1.text_input("岗位名称", value="AI-HR 产品实习生 / AI 招聘工具产品经理")
        location = col2.text_input("工作地点", value="上海 / 远程协作")
        salary = col3.text_input("薪资范围", value="面议")
        col1, col2, col3 = st.columns(3)
        industry = col1.text_input("行业方向", value="AI + HR SaaS")
        company_stage = col2.text_input("公司阶段", value="早期产品验证阶段")
        priority = col3.selectbox("招聘优先级", ["高", "中", "低"])
        jd_text = st.text_area("粘贴完整岗位描述", key="jd_text", height=300)
        extra_context = st.text_area("公司卖点 / 团队亮点 / 排除项补充", height=100)
        parse_clicked = st.form_submit_button("AI 解析 JD")

    if parse_clicked:
        prompt = json.dumps(
            {
                "role_title": role_title,
                "location": location,
                "salary": salary,
                "industry": industry,
                "company_stage": company_stage,
                "priority": priority,
                "jd_text": jd_text,
                "extra_context": extra_context,
            },
            ensure_ascii=False,
            indent=2,
        )
        profile = call_deepseek(
            prompt,
            system_prompt=JD_SYSTEM_PROMPT,
            response_type="jd",
            context={"role_title": role_title, "location": location},
        )
        st.session_state.job_profile_draft = profile
        add_audit_log(st.session_state.audit_log, "JD 解析", f"解析岗位：{role_title}")
        st.success("已生成结构化岗位画像。")

    profile = st.session_state.job_profile_draft
    if profile:
        st.subheader("结构化岗位画像")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(
                f"""
                <div class="tm-card">
                  <div class="tm-card-title">{escape(str(profile.get('role_title', '未明确岗位')))}</div>
                  <div class="tm-card-meta">
                    {escape(str(profile.get('location', '未明确')))} · {escape(str(profile.get('seniority_level', '未明确')))}
                  </div>
                  <div style="margin-top:10px">{badge('岗位画像已生成', 'good')}{badge('信息不足处标记为未明确', 'warn' if '未明确' in json.dumps(profile, ensure_ascii=False) else 'good')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("核心职责", expanded=True):
                st.markdown("\n".join(f"- {item}" for item in profile.get("responsibilities", [])) or "未明确")
            with st.expander("理想候选人画像", expanded=True):
                st.write(profile.get("ideal_candidate_profile", "未明确"))
        with col2:
            with st.expander("硬性要求", expanded=True):
                st.markdown("\n".join(f"- {item}" for item in profile.get("must_have_requirements", [])) or "未明确")
            with st.expander("加分项", expanded=True):
                st.markdown("\n".join(f"- {item}" for item in profile.get("nice_to_have_requirements", [])) or "未明确")
            with st.expander("评分权重", expanded=True):
                st.write(profile.get("scoring_weights", {}))
        with st.expander("结构化 JSON（调试视图）", expanded=False):
            st.json(profile)
        if st.button("保存岗位"):
            seed = json.dumps(profile, ensure_ascii=False) + now_text()
            job = {
                "job_id": stable_id("J", seed),
                "created_at": now_text(),
                "raw_jd": st.session_state.jd_text,
                "profile": profile,
            }
            st.session_state.jobs.append(job)
            add_audit_log(st.session_state.audit_log, "保存岗位", f"保存岗位：{job_label(job)}")
            st.success(f"已保存岗位：{job_label(job)}")


def load_sample_candidates() -> list[dict[str, Any]]:
    path = Path(__file__).with_name("sample_candidates.csv")
    df = pd.read_csv(path, keep_default_na=False)
    return [ensure_candidate_defaults(row) for row in df.to_dict(orient="records")]


def seed_mock_candidates_if_empty() -> int:
    if st.session_state.candidates or st.session_state.get("mock_candidates_seeded"):
        return 0
    candidates = load_sample_candidates()
    st.session_state.candidates = candidates
    st.session_state.mock_candidates_seeded = True
    add_audit_log(
        st.session_state.audit_log,
        "自动载入 Mock 候选人",
        f"首次进入会话自动载入 {len(candidates)} 条 mock 候选人。",
    )
    return len(candidates)


def page_candidate_import() -> None:
    render_header("导入候选人", "只接收合规来源资料：手动粘贴、CSV / Excel、企业授权人才库或官方 API 占位。")
    render_workflow_strip(active=1)
    render_trust_panel()
    st.info(
        f"已自动载入 {len(st.session_state.candidates)} 条 mock 候选人作为演示样本；资料仍只来自手动导入、CSV / Excel、授权通道或 mock source。"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("补齐 / 重新加载 100 条 mock 候选人"):
            warnings = add_candidates(load_sample_candidates(), "导入示例候选人")
            st.success(f"已从 sample_candidates.csv 补齐示例候选人；当前候选人池 {len(st.session_state.candidates)} 人。")
            for item in warnings:
                st.warning(item)

        uploaded = st.file_uploader("上传 CSV / Excel 候选人列表", type=["csv", "xlsx"])
        if uploaded:
            if uploaded.name.lower().endswith(".xlsx"):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)
            rows = []
            for row in df.to_dict(orient="records"):
                row["source_type"] = row.get("source_type") or "csv_upload"
                rows.append(ensure_candidate_defaults(row))
            warnings = add_candidates(rows, "上传候选人文件")
            st.success(f"已读取 {len(rows)} 条候选人记录。")
            for item in warnings:
                st.warning(item)

    with col2:
        st.markdown("**官方 / 企业授权通道占位**")
        st.caption("当前 Demo 不真实连接脉脉。未来只能通过官方 API、企业授权人才库或候选人主动分享入口接收资料。")
        st.code("source_type = official_api_placeholder", language="python")

    st.subheader("粘贴候选人主页文本")
    pasted_profile = st.text_area("候选人公开/授权资料文本", height=130)
    if st.button("从粘贴文本创建候选人"):
        if pasted_profile.strip():
            warnings = add_candidates([build_candidate_from_profile_text(pasted_profile)], "粘贴候选人资料")
            st.success("已创建候选人记录，可在表格中继续补充字段。")
            for item in warnings:
                st.warning(item)
        else:
            st.warning("请先粘贴候选人资料。")

    st.subheader("手动录入候选人")
    with st.form("manual_candidate_form"):
        col1, col2, col3 = st.columns(3)
        name = col1.text_input("姓名")
        company = col2.text_input("当前公司")
        title = col3.text_input("当前职位")
        col1, col2, col3 = st.columns(3)
        location = col1.text_input("地点")
        years = col2.text_input("年限")
        source_type = col3.selectbox("来源类型", SOURCE_TYPES, index=0)
        skills = st.text_input("技能关键词")
        education = st.text_input("教育背景")
        work_exp = st.text_area("工作经历", height=80)
        project_exp = st.text_area("项目经历", height=80)
        profile_url = st.text_input("公开主页 URL")
        notes = st.text_area("备注", height=70)
        submitted = st.form_submit_button("添加候选人")

    if submitted:
        candidate = ensure_candidate_defaults(
            {
                "name": name or "待补充候选人",
                "current_company": company or "未明确",
                "current_title": title or "未明确",
                "location": location or "未明确",
                "years_of_experience": years or "未明确",
                "skills": skills,
                "education": education or "未明确",
                "work_experience": work_exp,
                "project_experience": project_exp,
                "public_profile_text": " ".join([work_exp, project_exp, skills]),
                "profile_url": profile_url,
                "source_type": source_type,
                "source_note": "招聘人员手动录入",
                "notes": notes,
            }
        )
        warnings = add_candidates([candidate], "手动录入候选人")
        st.success("已添加候选人。")
        for item in warnings:
            st.warning(item)

    st.subheader("候选人列表")
    if st.session_state.candidates:
        st.caption(f"当前候选人池：{len(st.session_state.candidates)} 人。卡片预览仅展示前 6 人，完整数据请展开表格视图。")
        preview = st.session_state.candidates[:6]
        card_cols = st.columns(2)
        for index, candidate in enumerate(preview):
            with card_cols[index % 2]:
                render_candidate_card(candidate, st.session_state.match_results.get(candidate.get("candidate_id")))
        with st.expander("表格视图（用于检查字段和导出前核对）", expanded=False):
            render_readable_table(dataframe_from_candidates(st.session_state.candidates), height=420)
    else:
        st.warning("暂无候选人，请加载示例数据或手动导入。")


def apply_candidate_filters(
    candidates: list[dict[str, Any]],
    match_results: dict[str, dict[str, Any]] | None = None,
    city: str = "",
    company: str = "",
    title: str = "",
    skill: str = "",
    statuses: list[str] | None = None,
    recommendations: list[str] | None = None,
    include_non_contactable: bool = True,
) -> list[dict[str, Any]]:
    match_results = match_results or {}
    statuses = statuses or []
    recommendations = recommendations or []
    result = []
    for candidate in candidates:
        match = match_results.get(candidate.get("candidate_id"), {})
        guarded = candidate.get("blacklist_status") == "是" or candidate.get("contact_status") in ["黑名单", "不再联系"]
        if not include_non_contactable and guarded:
            continue
        if city and city not in str(candidate.get("location", "")):
            continue
        if company and company not in str(candidate.get("current_company", "")):
            continue
        if title and title not in str(candidate.get("current_title", "")):
            continue
        if skill and skill.lower() not in str(candidate.get("skills", "")).lower():
            continue
        if statuses and candidate.get("contact_status") not in statuses:
            continue
        if recommendations and match.get("recommendation_level") not in recommendations:
            continue
        result.append(candidate)
    return result


def filter_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with st.expander("筛选候选人", expanded=True):
        col1, col2, col3 = st.columns(3)
        city = col1.text_input("城市筛选")
        company = col2.text_input("当前公司筛选")
        title = col3.text_input("职位关键词")
        col1, col2, col3 = st.columns(3)
        skill = col1.text_input("技能关键词")
        status = col2.multiselect("触达状态", CRM_STATUSES)
        recommendation = col3.multiselect("推荐等级", RECOMMENDATION_LEVELS)
        include_non_contactable = st.checkbox(
            "评分时包含黑名单 / 不再联系候选人（仅分析，不允许触达）",
            value=True,
        )
        st.caption("关闭后会从评分范围中排除黑名单 / 不再联系；开启时只参与分析，触达按钮仍保持禁用。")

    return apply_candidate_filters(
        candidates,
        st.session_state.match_results,
        city=city,
        company=company,
        title=title,
        skill=skill,
        statuses=status,
        recommendations=recommendation,
        include_non_contactable=include_non_contactable,
    )


def page_matching() -> None:
    render_header("评估候选人", "用证据、置信度和风险提示辅助 HR 判断，而不是只给一个神秘分数。")
    render_workflow_strip(active=1)
    if not st.session_state.jobs:
        st.warning("请先在“岗位创建”页面保存岗位。")
        return
    if not st.session_state.candidates:
        st.warning("请先在“候选人导入”页面加载或录入候选人。")
        return

    job_options = {job_label(job): job.get("job_id") for job in st.session_state.jobs}
    selected_job_label = st.selectbox("选择岗位", list(job_options.keys()))
    job = find_job(job_options[selected_job_label])
    assert job is not None

    filtered = filter_candidates(st.session_state.candidates)
    st.caption(f"当前筛选范围：{len(filtered)} / {len(st.session_state.candidates)} 人")
    st.info(f"本次将评分 {len(filtered)} 人；下方仅预览前 8 人，点击评分会对当前筛选范围全量分析。")
    if filtered:
        with st.expander("当前评估范围", expanded=True):
            card_cols = st.columns(2)
            for index, candidate in enumerate(filtered[:8]):
                with card_cols[index % 2]:
                    render_candidate_card(candidate, st.session_state.match_results.get(candidate.get("candidate_id")))

    if st.button("开始 AI 匹配评分", disabled=not filtered):
        progress = st.progress(0)
        for index, candidate in enumerate(filtered, start=1):
            prompt = json.dumps(
                {
                    "job_profile": job.get("profile"),
                    "candidate": candidate,
                    "instruction": "请根据岗位和候选人资料输出结构化匹配评分。",
                },
                ensure_ascii=False,
                indent=2,
            )
            result = call_deepseek(
                prompt,
                system_prompt=MATCH_SYSTEM_PROMPT,
                response_type="match",
                context={"job_profile": job.get("profile"), "candidate": candidate},
            )
            result = normalize_match_result(result, job.get("profile", {}), candidate)
            st.session_state.match_results[candidate.get("candidate_id")] = result
            st.session_state.match_history.insert(
                0,
                {
                    "time": now_text(),
                    "job_id": job.get("job_id"),
                    "candidate_id": candidate.get("candidate_id"),
                    "match_score": result.get("match_score"),
                    "recommendation_level": result.get("recommendation_level"),
                },
            )
            progress.progress(index / len(filtered))
        add_audit_log(st.session_state.audit_log, "AI 匹配评分", f"岗位 {job.get('job_id')} 评分 {len(filtered)} 人。")
        st.success("匹配评分已完成。")

    results = [
        st.session_state.match_results.get(candidate.get("candidate_id"))
        for candidate in filtered
        if candidate.get("candidate_id") in st.session_state.match_results
    ]
    results = [item for item in results if item]
    if results:
        st.subheader("证据级匹配结果")
        results = sorted(results, key=lambda item: int(item.get("match_score", 0) or 0), reverse=True)
        for item in results:
            candidate = find_candidate(item.get("candidate_id"))
            if candidate:
                render_candidate_card(candidate, item)
            with st.expander(f"查看证据与面试关注点｜{item.get('candidate_name')}"):
                evidence_df = pd.DataFrame(item.get("evidence", []))
                if not evidence_df.empty:
                    st.dataframe(evidence_df, use_container_width=True, hide_index=True, height=220)
                st.write("匹配理由：", item.get("match_reasons", []))
                st.write("风险点：", item.get("risks", []))
                st.write("缺失信息：", item.get("missing_information", []))
                st.write("面试关注点：", item.get("interview_focus", []))
                st.write("建议下一步：", item.get("suggested_next_action", "人工复核"))
        with st.expander("表格视图（调试与复核）", expanded=False):
            render_readable_table(pd.DataFrame(results), height=420)


def page_candidate_detail() -> None:
    render_header("候选人详情", "围绕一个候选人做判断：看证据、看风险、看合规状态，再决定是否生成草稿。")
    if not st.session_state.candidates:
        st.warning("暂无候选人。")
        return

    options = {candidate_label(candidate): candidate.get("candidate_id") for candidate in st.session_state.candidates}
    selected = st.selectbox("选择候选人", list(options.keys()))
    candidate = find_candidate(options[selected])
    assert candidate is not None
    can_outreach = candidate.get("blacklist_status") != "是" and candidate.get("contact_status") not in ["黑名单", "不再联系"]
    match = st.session_state.match_results.get(candidate.get("candidate_id"))
    render_workflow_strip(active=2 if match else 1)
    render_trust_panel(candidate, len(st.session_state.outreach_history))
    render_candidate_card(candidate, match)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"""
            <div class="tm-card">
              <div class="tm-card-title">候选人概览</div>
              <div class="tm-card-meta">ID：{escape(str(candidate.get('candidate_id')))}</div>
              <div style="margin-top:10px">
                {badge('公司：' + str(candidate.get('current_company', '未明确')), 'info')}
                {badge('职位：' + str(candidate.get('current_title', '未明确')), 'info')}
                {badge('地点：' + str(candidate.get('location', '未明确')), '')}
                {badge('年限：' + str(candidate.get('years_of_experience', '未明确')), '')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        sensitive_findings = detect_sensitive_terms(candidate.get("public_profile_text", ""))
        if sensitive_findings:
            st.warning(f"资料中出现可能敏感类别：{', '.join(sensitive_findings)}。系统不会基于这些内容评分或生成判断。")
        elif can_outreach:
            st.success("当前候选人未命中黑名单 / 不再联系状态。")
        else:
            st.error("该候选人处于黑名单或不再联系状态，默认禁止生成触达。")

    with st.expander("完整候选人资料（调试视图）", expanded=False):
        st.json(candidate)

    if match:
        with st.expander("AI 匹配证据", expanded=True):
            metric_cols = st.columns(3)
            metric_cols[0].metric("匹配分", match.get("match_score", 0), match.get("recommendation_level", ""))
            metric_cols[1].metric("置信度", match.get("confidence_level", "待评估"))
            metric_cols[2].metric("评分模式", "演示评分" if match.get("_mock") else "DeepSeek")
            evidence_df = pd.DataFrame(match.get("evidence", []))
            if not evidence_df.empty:
                st.dataframe(evidence_df, use_container_width=True, hide_index=True, height=220)
            st.write("匹配理由：", match.get("match_reasons", []))
            st.write("风险点：", match.get("risks", []))
            st.write("缺失信息：", match.get("missing_information", []))
            st.write("面试关注点：", match.get("interview_focus", []))
    else:
        st.info("该候选人尚未生成匹配评分。")

    if not st.session_state.jobs:
        st.warning("请先保存岗位，才能生成触达话术。")
        return

    job_options = {job_label(job): job.get("job_id") for job in st.session_state.jobs}
    selected_job = st.selectbox("选择触达岗位", list(job_options.keys()), key="detail_job")
    job = find_job(job_options[selected_job])
    style = st.selectbox("话术风格", ["专业正式", "轻松自然", "猎头风格"])

    if st.button("生成触达话术草稿", disabled=not can_outreach):
        prompt = json.dumps(
            {
                "job_profile": job.get("profile") if job else {},
                "candidate": candidate,
                "match_result": match or {},
                "message_style": style,
            },
            ensure_ascii=False,
            indent=2,
        )
        draft = call_deepseek(
            prompt,
            system_prompt=OUTREACH_SYSTEM_PROMPT,
            response_type="outreach",
            context={
                "job_profile": job.get("profile") if job else {},
                "candidate": candidate,
                "match_result": match or {},
                "message_style": style,
            },
        )
        draft["_copied_to_external_channel"] = False
        st.session_state.outreach_drafts[candidate.get("candidate_id")] = draft
        update_candidate(candidate.get("candidate_id"), {"contact_status": "已生成话术"})
        add_audit_log(st.session_state.audit_log, "生成触达草稿", f"{candidate.get('candidate_id')} 生成 {style} 话术。")
        st.success("已生成触达草稿。")

    draft = st.session_state.outreach_drafts.get(candidate.get("candidate_id"))
    if draft:
        with st.expander("触达话术草稿", expanded=True):
            st.markdown(f"<div class='tm-copy'>{escape(str(draft.get('message_draft', '')))}</div>", unsafe_allow_html=True)
            st.write("个性化依据：", draft.get("personalization_points", []))
            st.write("合规检查：", draft.get("compliance_check", {}))
            st.info("Demo 只生成草稿，不真实发送；HR 复核后可复制到官方 App 或授权渠道。")
            if st.button("保存为待人工确认"):
                update_candidate(candidate.get("candidate_id"), {"contact_status": "待人工确认"})
                st.session_state.outreach_history.insert(
                    0,
                    {
                        "time": now_text(),
                        "candidate_id": candidate.get("candidate_id"),
                        "style": draft.get("message_style"),
                        "action": "保存为待人工确认",
                    },
                )
                add_audit_log(st.session_state.audit_log, "人工确认待办", f"{candidate.get('candidate_id')} 保存为待人工确认。")
                st.success("已进入待人工确认状态。")


def page_crm() -> None:
    render_header("确认触达", "最后一步只做人工确认、状态管理和合规导出；系统不替 HR 发送消息。")
    render_workflow_strip(active=2)
    if not st.session_state.candidates:
        st.warning("暂无候选人。")
        return

    card_cols = st.columns(2)
    for index, item in enumerate(st.session_state.candidates[:6]):
        with card_cols[index % 2]:
            render_candidate_card(item, st.session_state.match_results.get(item.get("candidate_id")))
    with st.expander("候选人表格视图", expanded=False):
        render_readable_table(dataframe_from_candidates(st.session_state.candidates), height=420)

    st.subheader("合规导出")
    export_confirmed = st.checkbox("我确认导出文件不包含 API Key、敏感属性推断、未授权数据或自动发送记录。")
    csv_data = dataframe_for_export(st.session_state.candidates)
    st.download_button(
        "导出候选人 CSV",
        csv_data.encode("utf-8-sig"),
        file_name="talentmatch_candidates_export.csv",
        mime="text/csv",
        disabled=not export_confirmed,
    )

    options = {candidate_label(candidate): candidate.get("candidate_id") for candidate in st.session_state.candidates}
    selected = st.selectbox("选择候选人进行管理", list(options.keys()))
    candidate = find_candidate(options[selected])
    assert candidate is not None
    render_trust_panel(candidate, len(st.session_state.outreach_history))

    col1, col2, col3 = st.columns(3)
    status_index = CRM_STATUSES.index(candidate.get("contact_status")) if candidate.get("contact_status") in CRM_STATUSES else 0
    new_status = col1.selectbox("候选人状态", CRM_STATUSES, index=status_index)
    next_action = col2.text_input("下一步行动", value=candidate.get("next_action", ""))
    reminder = col3.date_input("提醒时间", value=date.today())
    notes = st.text_area("备注", value=candidate.get("notes", ""), height=120)

    if st.button("保存 CRM 更新"):
        updates = {
            "contact_status": new_status,
            "notes": notes,
            "next_action": next_action,
            "reminder_time": str(reminder),
            "blacklist_status": "是" if new_status == "黑名单" else candidate.get("blacklist_status", "否"),
        }
        update_candidate(candidate.get("candidate_id"), updates)
        add_audit_log(st.session_state.audit_log, "CRM 状态更新", f"{candidate.get('candidate_id')} 更新为 {new_status}。")
        st.success("已保存 CRM 更新。")

    col1, col2, col3 = st.columns(3)
    if col1.button("标记不再联系"):
        update_candidate(candidate.get("candidate_id"), {"contact_status": "不再联系"})
        add_audit_log(st.session_state.audit_log, "不再联系", f"{candidate.get('candidate_id')} 标记不再联系。")
        st.warning("已标记为不再联系。")
    if col2.button("加入黑名单"):
        update_candidate(candidate.get("candidate_id"), {"contact_status": "黑名单", "blacklist_status": "是"})
        add_audit_log(st.session_state.audit_log, "黑名单", f"{candidate.get('candidate_id')} 加入黑名单。")
        st.error("已加入黑名单。")
    if col3.button("删除候选人数据"):
        candidate_id = candidate.get("candidate_id")
        st.session_state.candidates = [item for item in st.session_state.candidates if item.get("candidate_id") != candidate_id]
        st.session_state.match_results.pop(candidate_id, None)
        st.session_state.outreach_drafts.pop(candidate_id, None)
        add_audit_log(st.session_state.audit_log, "删除候选人", f"{candidate_id} 已从本地 session 删除。")
        st.success("已删除候选人数据。")
        st.rerun()

    draft = st.session_state.outreach_drafts.get(candidate.get("candidate_id"))
    with st.expander("触达历史与话术草稿", expanded=True):
        if draft:
            st.write(draft.get("message_draft"))
            st.write(draft.get("suggested_follow_up"))
        else:
            st.caption("暂无话术草稿。")
        history = [item for item in st.session_state.outreach_history if item.get("candidate_id") == candidate.get("candidate_id")]
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True, height=260)

    with st.expander("AI 匹配历史", expanded=False):
        history = [item for item in st.session_state.match_history if item.get("candidate_id") == candidate.get("candidate_id")]
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True, height=260)
        else:
            st.caption("暂无匹配历史。")


def page_compliance() -> None:
    render_header("来源与授权中心", "把自动检索来源、授权状态、数据边界和审计能力做成后台护栏，不打断主流程。")
    render_trust_panel()
    st.markdown(
        """
        <div class="tm-source-strip">
          <div class="tm-source-item">
            <div class="tm-source-name">当前模式</div>
            <div class="tm-source-desc">Mock source + 本地人才库，保证无外部网络也能完整演示。</div>
          </div>
          <div class="tm-source-item">
            <div class="tm-source-name">授权策略</div>
            <div class="tm-source-desc">未来只接官方 API、企业授权或候选人主动分享数据。</div>
          </div>
          <div class="tm-source-item">
            <div class="tm-source-name">触达策略</div>
            <div class="tm-source-desc">只生成草稿，不自动发送，关键动作必须人工确认。</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    principles = [
        ("已启用来源", ["mock_search", "local_talent_pool", "CSV / Excel 上传", "手动粘贴"]),
        ("预留来源", ["ATS", "官方搜索 API", "招聘平台授权 API", "official_api_placeholder"]),
        ("不可用能力", ["非官方抓取", "绕过验证码", "自动群发私信", "保存 API Key"]),
        ("审计状态", ["导入记录", "检索记录", "评分记录", "导出记录"]),
    ]
    for col, (title, items) in zip(cols, principles):
        col.markdown(
            f"""
            <div class="tm-card">
              <div class="tm-card-title">{escape(title)}</div>
              <ul>{html_list(items, 6)}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="tm-card">
          <div class="tm-card-title">Connector 边界</div>
          <div class="tm-card-meta">
            <code>mock_search</code> 用于现场演示自动发现候选人；<code>official_api_placeholder</code>
            表示未来通过官方 API、企业授权或候选人主动分享入口接收资料。
            当前 MVP 不实现非官方平台抓取或自动发送。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("面试表达建议：自动检索不是浏览器爬取，而是搜索策略生成 + 可控数据源召回 + HR 复核入库。")

    with st.expander("审计日志", expanded=False):
        if st.session_state.audit_log:
            st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True, hide_index=True, height=320)
        else:
            st.caption("暂无审计日志。")


def main() -> None:
    setup_page()
    init_state()
    seed_mock_candidates_if_empty()
    page = sidebar()

    if page == "今日驾驶舱":
        page_dashboard()
    elif page == "0 自动检索":
        page_auto_sourcing()
    elif page == "1 定义岗位":
        page_job_creation()
    elif page == "2 导入候选人":
        page_candidate_import()
    elif page == "3 评估候选人":
        page_matching()
    elif page == "候选人详情":
        page_candidate_detail()
    elif page == "确认触达":
        page_crm()
    elif page == "来源与授权中心":
        page_compliance()


if __name__ == "__main__":
    main()
