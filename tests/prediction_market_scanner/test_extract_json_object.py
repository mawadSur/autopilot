"""Tests for the consolidated ``extract_json_object`` helper in ``src/utils.py``."""

from __future__ import annotations

import unittest

from utils import extract_json_object


class ExtractJsonObjectTests(unittest.TestCase):
    def test_valid_json_returns_dict(self) -> None:
        text = '{"clarity_score": 87, "narrative_momentum": 42, "ambiguous": false, "reasoning": "ok"}'
        parsed = extract_json_object(text)
        self.assertEqual(parsed["clarity_score"], 87)
        self.assertEqual(parsed["narrative_momentum"], 42)
        self.assertEqual(parsed["reasoning"], "ok")

    def test_markdown_fenced_json(self) -> None:
        text = '```json\n{"clarity_score": 70, "reasoning": "fenced"}\n```'
        parsed = extract_json_object(text)
        self.assertEqual(parsed["clarity_score"], 70)
        self.assertEqual(parsed["reasoning"], "fenced")

    def test_json_with_prose_preamble(self) -> None:
        text = (
            "Here is the analysis you requested:\n\n"
            '{"clarity_score": 55, "narrative_momentum": 60, "reasoning": "prose preamble case"}\n'
            "Hope that helps."
        )
        parsed = extract_json_object(text)
        self.assertEqual(parsed["clarity_score"], 55)
        self.assertEqual(parsed["narrative_momentum"], 60)
        self.assertIn("prose preamble", parsed["reasoning"])

    def test_malformed_json_falls_back_to_field_regex(self) -> None:
        # Closing brace missing — bracket extraction would fail too. Field-level
        # regex should still recover at least clarity_score.
        text = (
            "clarity_score: 33, narrative_momentum: 47, "
            'reasoning: "broken json but fields are present"'
        )
        parsed = extract_json_object(text)
        self.assertEqual(parsed["clarity_score"], 33)
        self.assertEqual(parsed["narrative_momentum"], 47)
        self.assertIn("broken json", parsed["reasoning"])

    def test_empty_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object("")

    def test_none_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object(None)  # type: ignore[arg-type]

    def test_completely_unrecoverable_input_raises(self) -> None:
        # No JSON, no recognizable field names — must raise.
        with self.assertRaises(ValueError):
            extract_json_object("the quick brown fox")

    def test_lowercase_fence(self) -> None:
        text = '```\n{"clarity_score": 12}\n```'
        parsed = extract_json_object(text)
        self.assertEqual(parsed["clarity_score"], 12)


if __name__ == "__main__":
    unittest.main()
