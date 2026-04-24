import unittest

from news_research_agent.analyzer import NewsAgent


class NewsAgentTests(unittest.TestCase):
    def test_analyze_news_builds_structured_report(self):
        agent = NewsAgent()
        report = agent.analyze_news(
            market_title="Will candidate X win?",
            implied_prob=0.42,
            news_context=(
                "NEWS COVERAGE CONTEXT\n"
                "ARTICLE 1 | 2026-04-19T09:00:00+00:00 | Article A\n"
                "Summary: Summary A\n"
                "Link: https://example.com/a\n\n"
                "ARTICLE 2 | 2026-04-20T12:00:00+00:00 | Article B\n"
                "Summary: Summary B\n"
            ),
        )

        self.assertEqual(report.timeline[0], "2026-04-19T09:00:00+00:00 | Article A")
        self.assertEqual(report.key_facts[0], "Summary A")
        self.assertGreaterEqual(report.source_quality_score, 4)
        self.assertIn("Google News RSS returned", report.summary)

    def test_invalid_inputs_raise(self):
        agent = NewsAgent()
        with self.assertRaisesRegex(ValueError, "market_title"):
            agent.analyze_news(market_title="", implied_prob=0.4, news_context="ctx")
        with self.assertRaisesRegex(ValueError, "news_context"):
            agent.analyze_news(market_title="x", implied_prob=0.4, news_context="")
        with self.assertRaisesRegex(ValueError, "implied_prob"):
            agent.analyze_news(market_title="x", implied_prob=1.4, news_context="ctx")


if __name__ == "__main__":
    unittest.main()
