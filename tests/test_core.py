import csv
from pathlib import Path
import unittest

from llm_client import mock_llm_response
from app import apply_candidate_filters, load_sample_candidates
from sourcing import (
    build_search_strategy,
    candidate_from_sourcing_result,
    profile_completeness,
    run_mock_search,
)
from utils import (
    build_compliance_status,
    build_candidate_from_profile_text,
    calculate_mock_score,
    dedupe_candidates,
    detect_sensitive_terms,
    next_best_action,
    recommendation_from_score,
)


class TalentMatchCoreTests(unittest.TestCase):
    def test_sample_candidates_contains_large_diverse_mock_pool(self):
        sample_path = Path(__file__).resolve().parents[1] / "sample_candidates.csv"
        with sample_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 100)
        self.assertTrue(any(not row.get("current_title") or not row.get("skills") for row in rows))
        self.assertTrue(
            any(row.get("current_title") in ["品牌运营", "销售经理"] or "纯销售" in row.get("skills", "") for row in rows)
        )
        self.assertTrue(any(row.get("blacklist_status") == "是" for row in rows))
        self.assertGreaterEqual(len({row.get("source_type") for row in rows}), 5)

    def test_sample_loader_returns_100_mock_candidates_without_nan_strings(self):
        candidates = load_sample_candidates()

        self.assertEqual(len(candidates), 100)
        self.assertTrue(any(not str(candidate.get("current_title", "")).strip() for candidate in candidates))
        self.assertFalse(any(str(value).lower() == "nan" for candidate in candidates for value in candidate.values()))

    def test_scoring_filter_includes_guarded_candidates_by_default(self):
        candidates = [
            {"candidate_id": "C001", "contact_status": "未触达", "blacklist_status": "否"},
            {"candidate_id": "C002", "contact_status": "不再联系", "blacklist_status": "否"},
            {"candidate_id": "C003", "contact_status": "黑名单", "blacklist_status": "是"},
        ]

        included = apply_candidate_filters(candidates, include_non_contactable=True)
        excluded = apply_candidate_filters(candidates, include_non_contactable=False)

        self.assertEqual([item["candidate_id"] for item in included], ["C001", "C002", "C003"])
        self.assertEqual([item["candidate_id"] for item in excluded], ["C001"])

    def test_dedupe_candidates_flags_repeated_identity(self):
        candidates = [
            {
                "candidate_id": "C001",
                "name": "王一",
                "current_company": "Alpha",
                "current_title": "AI 产品经理",
                "profile_url": "https://example.com/a",
            },
            {
                "candidate_id": "C001",
                "name": "王一",
                "current_company": "Alpha",
                "current_title": "AI 产品经理",
                "profile_url": "https://example.com/a",
            },
            {
                "candidate_id": "C009",
                "name": "王一",
                "current_company": "Alpha",
                "current_title": "AI 产品经理",
                "profile_url": "",
            },
        ]

        unique, warnings = dedupe_candidates(candidates)

        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0]["candidate_id"], "C001")
        self.assertGreaterEqual(len(warnings), 2)
        self.assertTrue(any("candidate_id" in item for item in warnings))

    def test_sensitive_terms_detects_disallowed_attributes(self):
        findings = detect_sensitive_terms("候选人年龄 35，已婚，党员，身体健康。")

        self.assertIn("年龄", findings)
        self.assertIn("婚育", findings)
        self.assertIn("政治", findings)
        self.assertIn("健康", findings)

    def test_build_candidate_from_profile_text_uses_safe_defaults(self):
        candidate = build_candidate_from_profile_text("候选人做过 HR SaaS 和 AI 招聘工具。")

        self.assertEqual(candidate["source_type"], "manual_input")
        self.assertEqual(candidate["contact_status"], "未触达")
        self.assertEqual(candidate["blacklist_status"], "否")
        self.assertIn("HR SaaS", candidate["public_profile_text"])

    def test_recommendation_from_score_boundaries(self):
        self.assertEqual(recommendation_from_score(86), "强烈推荐")
        self.assertEqual(recommendation_from_score(70), "可以联系")
        self.assertEqual(recommendation_from_score(50), "观望")
        self.assertEqual(recommendation_from_score(20), "不推荐")

    def test_mock_llm_response_returns_required_matching_shape(self):
        result = mock_llm_response(
            "请进行候选人匹配评分 candidate_id C001 name 王一 AI HR SaaS",
            response_type="match",
        )

        self.assertEqual(result["candidate_id"], "C001")
        self.assertIn("match_score", result)
        self.assertIn("score_breakdown", result)
        self.assertIn("match_reasons", result)
        self.assertIn("manual", result["suggested_next_action"].lower())

    def test_next_best_action_respects_contact_guardrails(self):
        candidate = {
            "candidate_id": "C010",
            "name": "李明",
            "current_title": "AI 产品经理",
            "contact_status": "不再联系",
            "blacklist_status": "否",
            "skills": "AI 产品",
        }

        self.assertEqual(next_best_action(candidate, {"match_score": 88}), "不再联系")

        candidate["contact_status"] = "未触达"
        self.assertEqual(next_best_action(candidate, {"match_score": 88}), "生成草稿")

        candidate["contact_status"] = "待人工确认"
        self.assertEqual(next_best_action(candidate, {"match_score": 88}), "人工确认")

    def test_compliance_status_summarizes_source_sensitive_and_contactability(self):
        candidate = {
            "source_type": "public_profile",
            "public_profile_text": "公开资料提到年龄和婚育，但这些不能用于判断。",
            "contact_status": "未触达",
            "blacklist_status": "否",
        }

        status = build_compliance_status(candidate, outreach_used=3, daily_limit=20)

        self.assertEqual(status["data_source"], "public_profile")
        self.assertEqual(status["contactable"], True)
        self.assertEqual(status["remaining_quota"], 17)
        self.assertIn("年龄", status["sensitive_findings"])
        self.assertIn("婚育", status["sensitive_findings"])

    def test_mock_score_includes_confidence_and_evidence(self):
        job_profile = {
            "must_have_requirements": ["AI 产品设计", "HR SaaS 产品"],
            "nice_to_have_requirements": ["Python", "Streamlit"],
            "responsibilities": ["候选人匹配", "触达草稿"],
            "company_selling_points": ["合规 AI 招聘工具"],
        }
        candidate = {
            "candidate_id": "C011",
            "name": "安然",
            "skills": "AI 产品设计; Python; Streamlit",
            "work_experience": "负责 HR SaaS 候选人匹配和触达草稿工具。",
            "project_experience": "搭建过合规 AI 招聘工具 Demo。",
            "public_profile_text": "AI 产品设计、HR SaaS、Python、Streamlit。",
            "location": "上海",
        }

        result = calculate_mock_score(job_profile, candidate)

        self.assertIn(result["confidence_level"], ["高", "中", "低"])
        self.assertGreaterEqual(len(result["evidence"]), 2)
        self.assertIn("job_requirement", result["evidence"][0])
        self.assertIn("candidate_evidence", result["evidence"][0])

    def test_build_search_strategy_creates_queries_and_variants(self):
        job_profile = {
            "role_title": "AI-HR 产品经理",
            "location": "上海",
            "must_have_requirements": ["AI 产品设计", "HR SaaS", "Python"],
            "nice_to_have_requirements": ["招聘运营", "数据分析"],
            "excluded_conditions": ["纯销售"],
        }

        strategy = build_search_strategy(job_profile)

        self.assertIn("AI 产品设计", strategy["keywords"])
        self.assertIn("上海", strategy["locations"])
        self.assertIn("AI-HR 产品经理", strategy["title_aliases"])
        self.assertIn("纯销售", strategy["excluded_keywords"])
        self.assertGreaterEqual(len(strategy["boolean_queries"]), 5)
        self.assertLessEqual(len(strategy["boolean_queries"]), 8)

    def test_mock_search_returns_sourcing_leads_with_metadata(self):
        strategy = build_search_strategy(
            {
                "role_title": "AI 招聘工具产品经理",
                "location": "上海",
                "must_have_requirements": ["AI 产品", "HR SaaS"],
                "nice_to_have_requirements": ["Python"],
            }
        )
        pool = [
            {
                "candidate_id": "C020",
                "name": "沈星",
                "current_company": "星云 HR",
                "current_title": "AI 产品经理",
                "location": "上海",
                "skills": "AI 产品; HR SaaS; Python",
                "work_experience": "负责 AI 招聘工具。",
                "project_experience": "做过候选人匹配。",
                "public_profile_text": "AI 产品 HR SaaS Python",
                "profile_url": "https://example.com/c020",
            }
        ]

        leads = run_mock_search(strategy, pool, limit=3)

        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["retrieval_source"], "mock_search")
        self.assertIn("retrieval_query", leads[0])
        self.assertEqual(leads[0]["sourcing_status"], "新发现")
        self.assertGreaterEqual(leads[0]["profile_completeness"], 70)

    def test_candidate_from_sourcing_result_is_ready_for_existing_pipeline(self):
        lead = {
            "candidate_id": "C021",
            "name": "顾青",
            "current_company": "Alpha",
            "current_title": "HR SaaS 产品经理",
            "location": "北京",
            "skills": "HR SaaS; 招聘 CRM",
            "public_profile_text": "HR SaaS 招聘 CRM",
            "retrieval_source": "mock_search",
            "retrieval_query": "HR SaaS AND 招聘 CRM",
        }

        candidate = candidate_from_sourcing_result(lead)

        self.assertEqual(candidate["source_type"], "mock_search")
        self.assertEqual(candidate["retrieval_source"], "mock_search")
        self.assertEqual(candidate["sourcing_status"], "待复核")
        self.assertEqual(candidate["contact_status"], "未触达")

    def test_profile_completeness_penalizes_sparse_profiles(self):
        sparse = {"name": "匿名", "skills": "", "public_profile_text": ""}
        rich = {
            "name": "许言",
            "current_company": "Beta",
            "current_title": "AI 产品经理",
            "location": "上海",
            "years_of_experience": "5",
            "skills": "AI 产品; Python",
            "education": "本科",
            "work_experience": "负责产品",
            "project_experience": "招聘工具",
            "public_profile_text": "AI 产品经理，做过招聘工具",
            "profile_url": "https://example.com/rich",
        }

        self.assertLess(profile_completeness(sparse), 40)
        self.assertGreaterEqual(profile_completeness(rich), 90)


if __name__ == "__main__":
    unittest.main()
