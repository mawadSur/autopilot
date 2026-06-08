import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import NarrativeAnalysis, SocialPost


class ModelSchemaTests(unittest.TestCase):
    def test_social_post_accepts_valid_payload(self) -> None:
        post = SocialPost(
            platform="x",
            author_id="user-123",
            text="Market chatter is turning euphoric.",
            is_reply=False,
            is_quote=True,
            linked_urls=["https://example.com/thread"],
            engagement_score=240,
        )

        self.assertEqual(post.platform, "x")
        self.assertEqual(len(post.linked_urls), 1)
        self.assertEqual(post.engagement_score, 240)

    def test_narrative_analysis_requires_exactly_five_beliefs(self) -> None:
        with self.assertRaises(ValidationError):
            NarrativeAnalysis(
                bullish_thesis="The crowd expects an upside breakout.",
                bearish_thesis="Macro data can still invalidate the rally.",
                unresolved_questions=["Will volume confirm the move?"],
                signal_quality_score=7,
                crowd_overconfidence_score=8,
                misinformation_risk=4,
                crowd_beliefs=["One", "Two", "Three", "Four"],
                market_alignment="ahead",
                reasoning="The narrative is heating up before price fully reflects it.",
            )

    def test_narrative_analysis_rejects_out_of_range_scores(self) -> None:
        with self.assertRaises(ValidationError):
            NarrativeAnalysis(
                bullish_thesis="The market is underpricing the catalyst.",
                bearish_thesis="The catalyst may already be stale.",
                unresolved_questions=["Is the source primary?"],
                signal_quality_score=11,
                crowd_overconfidence_score=5,
                misinformation_risk=3,
                crowd_beliefs=["A", "B", "C", "D", "E"],
                market_alignment="aligned",
                reasoning="Sentiment and price are moving together.",
            )


if __name__ == "__main__":
    unittest.main()
