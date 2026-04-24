from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyzer import NarrativeAnalyzer
from fetcher import SocialAggregator
from models import NarrativeAnalysis, SocialPost


RESET = "[0m"
BOLD = "[1m"
GREEN = "[32m"
RED = "[31m"
YELLOW = "[33m"
CYAN = "[36m"
DIM = "[2m"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the social narrative agent pipeline.")
    parser.add_argument("--topic", required=True, help="Topic to scan across social posts.")
    parser.add_argument(
        "--current-odds",
        required=True,
        type=float,
        help="Current market odds as a 0.0-1.0 float.",
    )
    return parser.parse_args(argv)


def _score_color(score: int) -> str:
    if score >= 7:
        return GREEN
    if score >= 4:
        return YELLOW
    return RED


def _risk_color(score: int) -> str:
    if score >= 7:
        return YELLOW
    if score >= 4:
        return CYAN
    return GREEN


def _alignment_color(alignment: str) -> str:
    normalized = str(alignment or "").strip().lower()
    if normalized == "ahead":
        return GREEN
    if normalized == "behind":
        return RED
    return CYAN


def _colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _format_score(label: str, score: int, *, risk: bool = False) -> str:
    color = _risk_color(score) if risk else _score_color(score)
    bar = "■" * score + "·" * (10 - score)
    return f"- **{label}:** {_colorize(f'{score}/10 {bar}', color)}"


def render_markdown_report(
    *,
    topic: str,
    current_odds: float,
    posts: Sequence[SocialPost],
    formatted_social_text: str,
    analysis: NarrativeAnalysis,
) -> str:
    bullish_heading = _colorize(f"{BOLD}Bullish Thesis{RESET}", GREEN)
    bearish_heading = _colorize(f"{BOLD}Bearish Thesis{RESET}", RED)
    alignment_badge = _colorize(analysis.market_alignment.upper(), _alignment_color(analysis.market_alignment))
    misinformation_label = _colorize(str(analysis.misinformation_risk), _risk_color(analysis.misinformation_risk))
    engaged_posts = sum(1 for post in posts if post.engagement_score > 0)

    belief_lines = "\n".join(f"- {belief}" for belief in analysis.crowd_beliefs)
    question_lines = "\n".join(f"- {question}" for question in analysis.unresolved_questions) or "- None"
    social_preview = "\n".join(formatted_social_text.splitlines()[:6]) or "No formatted social text available."

    return "\n".join(
        [
            f"# {_colorize(topic, BOLD)}",
            "",
            "## Market Snapshot",
            f"- **Current Odds:** {current_odds:.1%}",
            f"- **Fetched Posts:** {len(posts)} total / {engaged_posts} engaged",
            f"- **Market Alignment:** {alignment_badge}",
            "",
            "## Scorecard",
            _format_score("Signal Quality", analysis.signal_quality_score),
            _format_score("Crowd Overconfidence", analysis.crowd_overconfidence_score),
            _format_score("Misinformation Risk", analysis.misinformation_risk, risk=True),
            "",
            f"## {bullish_heading}",
            _colorize(analysis.bullish_thesis, GREEN),
            "",
            f"## {bearish_heading}",
            _colorize(analysis.bearish_thesis, RED),
            "",
            "## Crowd Beliefs",
            belief_lines,
            "",
            "## Unresolved Questions",
            question_lines,
            "",
            "## Alignment Reasoning",
            analysis.reasoning,
            "",
            "## Social Sample",
            f"{DIM}```text{RESET}",
            social_preview,
            f"{DIM}```{RESET}",
            "",
            f"{DIM}High misinformation risk marker: {misinformation_label}/10{RESET}",
        ]
    )


async def run_pipeline(
    *,
    topic: str,
    current_odds: float,
    aggregator: Optional[SocialAggregator] = None,
    analyzer: Optional[NarrativeAnalyzer] = None,
) -> str:
    social_aggregator = aggregator or SocialAggregator(topic)
    narrative_analyzer = analyzer or NarrativeAnalyzer()

    posts = social_aggregator.fetch_reddit_threads(topic=topic)
    formatted_social_text = social_aggregator.format_for_llm(posts)
    analysis = await narrative_analyzer.analyze_narrative(
        social_text=formatted_social_text,
        current_market_odds=current_odds,
    )
    return render_markdown_report(
        topic=topic,
        current_odds=current_odds,
        posts=posts,
        formatted_social_text=formatted_social_text,
        analysis=analysis,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(run_pipeline(topic=args.topic, current_odds=args.current_odds))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
