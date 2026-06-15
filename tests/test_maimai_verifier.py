import unittest

from maimai_verifier import (
    build_maimai_queries,
    extract_public_profile,
    normalize_search_results,
    parse_search_results_json,
    score_maimai_candidate,
    verify_maimai_candidate,
)


class MaimaiVerifierTests(unittest.TestCase):
    def test_build_maimai_queries_requires_name_and_keeps_stable_order(self):
        payload = {
            "candidate_name": "张三",
            "known_title": "产品经理",
            "known_company": "",
            "known_school": "复旦大学",
        }

        self.assertEqual(
            build_maimai_queries(payload),
            [
                "张三 脉脉",
                "site:maimai.cn 张三",
                "张三 产品经理 脉脉",
                "张三 复旦大学 脉脉",
            ],
        )

        with self.assertRaises(ValueError):
            build_maimai_queries({"candidate_name": " "})

    def test_normalize_search_results_filters_maimai_related_results(self):
        results = [
            {
                "title": "张三 - 产品经理 - 某科技公司 | 脉脉",
                "snippet": "复旦大学，AI HR SaaS",
                "link": "https://maimai.cn/profile/abc",
                "source_query": "张三 脉脉",
            },
            {
                "title": "张三的博客",
                "snippet": "个人博客",
                "url": "https://example.com/zhangsan",
            },
            {
                "name": "李四 | 脉脉",
                "description": "候选人页面",
                "url": "https://example.com/maimai-snapshot",
            },
        ]

        normalized = normalize_search_results(results)

        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["url"], "https://maimai.cn/profile/abc")
        self.assertEqual(normalized[0]["source_query"], "张三 脉脉")
        self.assertEqual(normalized[1]["title"], "李四 | 脉脉")

    def test_extract_public_profile_uses_snippet_and_manual_text_without_guessing(self):
        result = {
            "title": "张三 - 高级产品经理 - 某科技公司 | 脉脉",
            "snippet": "复旦大学，负责 AI HR SaaS 产品。",
            "url": "https://maimai.cn/profile/abc?from=search",
        }
        manual_text = "公开主页可见：张三，某科技公司，高级产品经理，毕业于复旦大学。"

        profile = extract_public_profile(result, manual_text)

        self.assertEqual(profile["name"], "张三")
        self.assertIn("高级产品经理", profile["title"])
        self.assertIn("某科技公司", profile["company"])
        self.assertIn("复旦大学", profile["school"])
        self.assertEqual(profile["profile_url"], "https://maimai.cn/profile/abc")
        self.assertEqual(profile["access_status"], "manual_public_text")

        sparse = extract_public_profile({"title": "脉脉用户", "snippet": "", "url": "https://maimai.cn/"})
        self.assertEqual(sparse["name"], "")
        self.assertEqual(sparse["company"], "")

    def test_score_maimai_candidate_uses_fixed_weight_model(self):
        payload = {
            "candidate_name": "张三",
            "known_title": "产品经理",
            "known_company": "某科技公司",
            "known_school": "复旦大学",
            "extra_keywords": "AI, HR SaaS",
        }
        profile = {
            "name": "张三",
            "title": "高级产品经理",
            "company": "某科技公司",
            "school": "复旦大学",
            "summary": "AI HR SaaS",
            "profile_url": "https://maimai.cn/profile/abc",
            "raw_text": "张三 高级产品经理 某科技公司 复旦大学 AI HR SaaS",
        }

        scored = score_maimai_candidate(payload, profile)

        self.assertEqual(scored["score_breakdown"], {
            "name": 40,
            "title": 20,
            "school": 20,
            "company": 15,
            "extra_keywords": 5,
        })
        self.assertEqual(scored["confidence_score"], 100)
        self.assertEqual(
            scored["matched_fields"],
            ["name", "title", "school", "company", "extra_keywords"],
        )
        self.assertTrue(scored["manual_review_required"])

    def test_verify_maimai_candidate_ranks_results_and_flags_access_limits(self):
        payload = {
            "candidate_name": "张三",
            "known_title": "产品经理",
            "known_company": "某科技公司",
            "known_school": "复旦大学",
            "extra_keywords": ["AI"],
        }
        search_results = [
            {
                "title": "张三 - 产品经理 - 某科技公司 | 脉脉",
                "snippet": "复旦大学，AI 产品。",
                "url": "https://maimai.cn/profile/best",
                "access_status": "login_required",
            },
            {
                "title": "张三 - 销售经理 - 其他公司 | 脉脉",
                "snippet": "公开摘要",
                "url": "https://maimai.cn/profile/weak",
            },
        ]

        result = verify_maimai_candidate(payload, search_results)

        self.assertEqual(result["best_match_url"], "https://maimai.cn/profile/best")
        self.assertGreaterEqual(result["confidence_score"], 80)
        self.assertTrue(result["manual_review_required"])
        self.assertIn("name", result["matched_fields"])
        self.assertEqual(result["candidates"][0]["access_status"], "access_limited")
        self.assertIn("HR", result["candidates"][0]["risk_note"])

    def test_verify_maimai_candidate_returns_null_best_match_below_threshold(self):
        payload = {
            "candidate_name": "王五",
            "known_title": "算法工程师",
            "known_company": "目标公司",
            "known_school": "清华大学",
        }
        search_results = [
            {
                "title": "张三 - 产品经理 | 脉脉",
                "snippet": "某科技公司",
                "url": "https://maimai.cn/profile/other",
            }
        ]

        result = verify_maimai_candidate(payload, search_results)

        self.assertIsNone(result["best_match_url"])
        self.assertEqual(result["confidence_score"], 0)
        self.assertTrue(result["manual_review_required"])
        self.assertIn("人工", result["risk_note"])

    def test_parse_search_results_json_accepts_list_or_wrapped_results(self):
        wrapped = '{"results": [{"title": "张三 | 脉脉", "url": "https://maimai.cn/profile/a"}]}'
        parsed, error = parse_search_results_json(wrapped)

        self.assertEqual(error, "")
        self.assertEqual(parsed[0]["title"], "张三 | 脉脉")

        parsed, error = parse_search_results_json("{bad json")

        self.assertEqual(parsed, [])
        self.assertIn("JSON", error)


if __name__ == "__main__":
    unittest.main()
