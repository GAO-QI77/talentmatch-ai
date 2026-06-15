import json
import subprocess
import sys
import unittest
from pathlib import Path

from openclaw_demo import build_demo_report, load_candidates, render_markdown


class OpenClawDemoTests(unittest.TestCase):
    def test_default_loader_reads_100_mock_candidates(self):
        candidates = load_candidates()

        self.assertEqual(len(candidates), 100)

    def test_build_demo_report_has_required_contract(self):
        report = build_demo_report(mode="mock", top_k=10)

        self.assertEqual(report["metadata"]["candidate_count"], 100)
        self.assertEqual(report["metadata"]["mode"], "mock")
        self.assertIn("job_profile", report)
        self.assertIn("search_strategy", report)
        self.assertIn("candidate_pool_summary", report)
        self.assertEqual(len(report["top_candidates"]), 10)
        self.assertGreater(len(report["search_strategy"]["boolean_queries"]), 0)

    def test_top_candidates_include_evidence_risks_and_next_action(self):
        report = build_demo_report(mode="mock", top_k=5)
        candidate = report["top_candidates"][0]

        self.assertIn("match_score", candidate)
        self.assertIn("evidence", candidate)
        self.assertIn("risks", candidate)
        self.assertIn("suggested_next_action", candidate)
        self.assertTrue(candidate["evidence"])

    def test_non_contactable_candidates_are_not_in_outreach_drafts(self):
        report = build_demo_report(mode="mock", top_k=100)
        draft_ids = {draft["candidate_id"] for draft in report["outreach_drafts"]}
        guarded_ids = {
            candidate["candidate_id"]
            for candidate in report["top_candidates"]
            if not candidate["contactable"]
        }

        self.assertFalse(draft_ids & guarded_ids)
        self.assertTrue(all(draft["confirmation_status"] == "待人工确认" for draft in report["outreach_drafts"]))
        self.assertTrue(all(draft["delivery_status"] == "未发送" for draft in report["outreach_drafts"]))

    def test_markdown_output_contains_interview_sections_without_secrets(self):
        report = build_demo_report(mode="mock", top_k=3)
        markdown = render_markdown(report)

        self.assertIn("一分钟产品介绍", markdown)
        self.assertIn("自动检索策略", markdown)
        self.assertIn("触达草稿示例", markdown)
        self.assertIn("未来迭代方向", markdown)
        self.assertNotIn("DEEPSEEK_API_KEY", markdown)
        self.assertNotIn(".env", markdown)

    def test_cli_json_output_is_valid_and_secret_safe(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "openclaw_demo.py",
                "--mode",
                "mock",
                "--top-k",
                "5",
                "--format",
                "json",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        parsed = json.loads(result.stdout)

        self.assertEqual(parsed["metadata"]["candidate_count"], 100)
        self.assertEqual(len(parsed["top_candidates"]), 5)
        self.assertNotIn("DEEPSEEK_API_KEY", result.stdout)
        self.assertNotIn(".env", result.stdout)


if __name__ == "__main__":
    unittest.main()
