---
name: talentmatch-ai
description: AI-HR sourcing and screening demo skill for OpenClaw. Use when the user wants to demonstrate a compliant recruiting copilot, parse a JD, source mock/local candidates, rank candidates, produce evidence-based shortlist recommendations, or generate outreach drafts that require human confirmation.
---

# TalentMatch AI OpenClaw Skill

TalentMatch AI is a safe AI-HR interview demo skill. It lets an OpenClaw agent run a reproducible recruiting workflow:

```text
JD → search strategy → local/mock candidate retrieval → ranking → evidence → outreach drafts → HR confirmation
```

## When To Use

Use this skill when the user asks to:

- Demonstrate an AI-HR sourcing copilot.
- Screen candidates for a JD.
- Generate a shortlist from local/mock candidates.
- Explain why candidates match a role.
- Produce outreach drafts for HR review.
- Show a safe OpenClaw agent workflow for an AI-HR interview.

Example user prompts:

- “用 TalentMatch AI 帮我演示一个 AI 产品经理岗位的候选人筛选闭环。”
- “Run the OpenClaw TalentMatch AI demo and give me the top 10 candidates.”
- “帮我生成 AI-HR 面试官能看的 sourcing agent 演示报告。”

## Safety Rules

This skill is a demo workflow, not an autonomous recruiting bot.

The agent must:

- Use only local mock data, CSV files, or user-provided JD text.
- Never log in to recruiting platforms.
- Never scrape pages, bypass platform controls, or solve CAPTCHAs.
- Never auto-send outreach messages.
- Never read, print, export, or summarize `.env` or API keys.
- Treat blacklist / do-not-contact candidates as analyzable but not contactable.
- Mark all outreach drafts as `待人工确认`.

If the user asks to contact candidates directly, respond that this demo can only generate drafts and next-action recommendations.

## Primary Command

Run the default interview demo:

```bash
python3 openclaw_demo.py --mode mock --top-k 10 --format markdown
```

Run JSON mode for downstream agent summarization:

```bash
python3 openclaw_demo.py --mode mock --top-k 5 --format json
```

Use a custom JD:

```bash
python3 openclaw_demo.py --mode mock --job-file path/to/jd.txt --top-k 10 --format markdown
```

Use a custom candidate CSV:

```bash
python3 openclaw_demo.py --mode mock --candidate-file path/to/candidates.csv --top-k 10 --format json
```

## Output Contract

The CLI returns either Markdown or JSON.

JSON contains:

- `metadata`
- `job_profile`
- `search_strategy`
- `candidate_pool_summary`
- `top_candidates`
- `outreach_drafts`
- `compliance_guardrails`
- `interview_talk_track`
- `future_iteration_notes`

Use `top_candidates` for shortlist explanation. Use `outreach_drafts` only as drafts requiring manual HR confirmation.

## Recommended OpenClaw Response Style

After running the command, summarize:

1. What the system did.
2. The top 3 candidates and why they match.
3. Any risks or missing information.
4. One outreach draft example.
5. The compliance guardrails.
6. How this evolves into V3 connector and V4 sourcing agent.

Keep the tone professional and interview-ready. Avoid claiming that real candidates were contacted.
