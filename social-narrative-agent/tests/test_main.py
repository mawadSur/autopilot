import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

main_spec = importlib.util.spec_from_file_location("social_narrative_agent_main", PROJECT_ROOT / "main.py")
if main_spec is None or main_spec.loader is None:
    raise RuntimeError("Unable to load social-narrative-agent main.py for tests")
main_module = importlib.util.module_from_spec(main_spec)
main_spec.loader.exec_module(main_module)

from models import NarrativeAnalysis, SocialPost

parse_args = main_module.parse_args
render_markdown_report = main_module.render_markdown_report
run_pipeline = main_module.run_pipeline


class FakeAggregator:
    def __init__(self):
        self.fetch_calls = []
        self.format_calls = []
        self.posts = [
            SocialPost(
                platform="reddit",
                author_id="alpha",
                text="Launch chatter is accelerating.",
                is_reply=False,
                is_quote=False,
                linked_urls=[],
                engagement_score=120,
            ),
            SocialPost(
                platform="reddit",
                author_id="bravo",
                text="No official confirmation yet.",
                is_reply=True,
                is_quote=False,
                linked_urls=[],
                engagement_score=18,
            ),
        ]

    def fetch_reddit_threads(self, topic=None, limit=50):
        self.fetch_calls.append({"topic": topic, "limit": limit})
        return list(self.posts)

    def format_for_llm(self, posts):
        self.format_calls.append(list(posts))
        return "TOPIC: GPT-5 release date\nT1 HOT | post | 120 | alpha | Launch chatter is accelerating."


class FakeAnalyzer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def analyze_narrative(self, social_text: str, current_market_odds: float):
        self.calls.append({"social_text": social_text, "current_market_odds": current_market_odds})
        return self.result


class MainTests(unittest.IsolatedAsyncioTestCase):
    def _analysis(self):
        return NarrativeAnalysis(
            bullish_thesis="Primary-source-adjacent chatter is outrunning the current odds.",
            bearish_thesis="The narrative may still be recycled speculation without confirmation.",
            unresolved_questions=["Is there a direct release date from the company?"],
            signal_quality_score=8,
            crowd_overconfidence_score=7,
            misinformation_risk=8,
            crowd_beliefs=[
                "A release window is near.",
                "Social conviction is increasing.",
                "The market may still be lagging.",
                "Rumor chains are amplifying certainty.",
                "Official confirmation is still missing.",
            ],
            market_alignment="ahead",
            reasoning="The crowd is leaning harder than the current odds because chatter is concentrating around a near-term catalyst.",
        )

    async def test_run_pipeline_fetches_formats_and_analyzes(self):
        aggregator = FakeAggregator()
        analyzer = FakeAnalyzer(self._analysis())

        report = await run_pipeline(
            topic="GPT-5 release date",
            current_odds=0.42,
            aggregator=aggregator,
            analyzer=analyzer,
        )

        self.assertEqual(aggregator.fetch_calls[0]["topic"], "GPT-5 release date")
        self.assertEqual(len(aggregator.format_calls[0]), 2)
        self.assertEqual(analyzer.calls[0]["current_market_odds"], 0.42)
        self.assertIn("# ", report)
        self.assertIn("Bullish Thesis", report)
        self.assertIn("Market Alignment", report)
        self.assertIn("Crowd Beliefs", report)

    def test_render_markdown_report_includes_scores_beliefs_and_alignment(self):
        posts = [
            SocialPost(
                platform="reddit",
                author_id="alpha",
                text="Launch chatter is accelerating.",
                is_reply=False,
                is_quote=False,
                linked_urls=[],
                engagement_score=120,
            )
        ]
        report = render_markdown_report(
            topic="GPT-5 release date",
            current_odds=0.37,
            posts=posts,
            formatted_social_text="TOPIC: GPT-5 release date\nT1 HOT | post | 120 | alpha | Launch chatter is accelerating.",
            analysis=self._analysis(),
        )

        self.assertIn("Signal Quality", report)
        self.assertIn("Crowd Overconfidence", report)
        self.assertIn("Misinformation Risk", report)
        self.assertIn("- A release window is near.", report)
        self.assertIn("AHEAD", report)
        self.assertIn("```text", report)
        self.assertIn("[32m", report)
        self.assertIn("[31m", report)
        self.assertIn("[33m", report)

    def test_parse_args_requires_topic_and_current_odds(self):
        args = parse_args(["--topic", "Election winner", "--current-odds", "0.61"])
        self.assertEqual(args.topic, "Election winner")
        self.assertEqual(args.current_odds, 0.61)


if __name__ == "__main__":
    unittest.main()
