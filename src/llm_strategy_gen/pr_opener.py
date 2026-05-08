"""PROpener — fourth stage of the E3 loop (SKELETON).

Given a :class:`FeatureProposal` that cleared the
:class:`BacktestRunner` Sharpe gate, materialize the proposed feature
on disk and (in production) open a draft PR via the ``gh`` CLI.

What this skeleton does
-----------------------
* Slugifies ``proposal.name`` to a safe filename.
* Writes the proposal's ``python_code`` to
  ``src/llm_strategy_gen/proposed_features/<slug>.py`` with a header
  comment that reproduces the LLM-supplied description, expected lift,
  risk notes, and the security caveats.
* Writes a stub test file to
  ``tests/prediction_market_scanner/test_llm_proposed_<slug>.py`` —
  deliberately stubbed (``self.skipTest("LLM-proposed feature; review
  required")``) so the test suite doesn't auto-run untrusted code.
* In ``dry_run=True`` mode (default), returns a dict describing the
  ``gh pr create`` invocation that *would* have been executed and the
  paths it wrote, but does NOT shell out to ``gh`` or ``git``.

What this skeleton does NOT do (TBD in follow-up PRs)
-----------------------------------------------------
* Real worktree / branch creation. Production deployment must land in
  a fresh worktree (see ``using-git-worktrees`` skill in the harness)
  or, more conservatively, an isolated CI worker with no commit
  authority on ``main``.
* Real ``gh pr create`` invocation. The skeleton stops at building the
  command string. Wiring it up means making the call, capturing the PR
  URL from stdout, and threading auth via ``GH_TOKEN``.
* Auto-rollback on failure. The skeleton never runs the proposed code,
  so there is nothing to roll back; once real backtest retraining is
  added, a failure path that nukes the branch will be needed.

ALL OPENED PRs MUST BE OPERATOR-REVIEWED. The system never auto-merges.
This is enforced by ``--draft`` plus the explicit caveat in the PR body.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_strategy_gen.backtest_runner import BacktestResult
from llm_strategy_gen.feature_proposal import FeatureProposal

LOGGER = logging.getLogger(__name__)


# Default location for the proposed-feature artifacts. Test runs
# override this to a tmpdir.
DEFAULT_FEATURES_DIR = Path("src/llm_strategy_gen/proposed_features")
DEFAULT_TESTS_DIR = Path("tests/prediction_market_scanner")
DEFAULT_BASE_BRANCH = "feature/prediction-market-bot"

# Regex for slugifying proposal names: replace any non-alnum/underscore
# with a single underscore; trim leading/trailing underscores; lowercase.
_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9_]+")


def _slugify(name: str) -> str:
    s = _SLUG_PATTERN.sub("_", str(name)).strip("_").lower()
    if not s:
        s = "unnamed_feature"
    # Cap length so the file path stays reasonable.
    return s[:48]


@dataclass(frozen=True)
class PROpenResult:
    """Structured result of a :meth:`PROpener.open_pr` call.

    Attributes
    ----------
    feature_path:
        Absolute path of the written feature file (always present, both
        dry and live runs).
    test_path:
        Absolute path of the written stub test file (always present).
    branch_name:
        Branch name we would have created. ``None`` in dry-run mode.
    pr_url:
        URL returned by ``gh pr create`` in live mode. ``None`` in
        dry-run mode or if the gh call fails.
    gh_command:
        The ``gh pr create ...`` argv list we would invoke. Always
        populated so operators can reproduce manually.
    dry_run:
        ``True`` if dry-run was requested.
    """

    feature_path: Path
    test_path: Path
    branch_name: Optional[str]
    pr_url: Optional[str]
    gh_command: List[str]
    dry_run: bool


class PROpener:
    """Materialize a passed proposal as files + (in prod) open a PR.

    Parameters
    ----------
    repo_root:
        Path to the repository root. The skeleton writes files
        underneath this; production wires up a worktree and points
        here.
    features_dir:
        Subpath under ``repo_root`` for the proposed feature files.
        Defaults to ``src/llm_strategy_gen/proposed_features``.
    tests_dir:
        Subpath under ``repo_root`` for stub test files. Defaults to
        ``tests/prediction_market_scanner``.
    base_branch:
        Base branch the PR would target. Defaults to
        ``feature/prediction-market-bot``.
    gh_runner:
        Optional callable
        ``(argv: List[str]) -> Dict[str, Any]`` for live PR creation.
        Returning a dict with ``url`` lets the opener record it. The
        skeleton never calls this in dry-run mode; production wires it
        to a ``subprocess.run``-based shim.
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        features_dir: Path = DEFAULT_FEATURES_DIR,
        tests_dir: Path = DEFAULT_TESTS_DIR,
        base_branch: str = DEFAULT_BASE_BRANCH,
        gh_runner: Optional[Any] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        # Allow features_dir to be either absolute or relative-to-repo-root.
        self.features_dir = (
            Path(features_dir)
            if Path(features_dir).is_absolute()
            else self.repo_root / Path(features_dir)
        )
        self.tests_dir = (
            Path(tests_dir)
            if Path(tests_dir).is_absolute()
            else self.repo_root / Path(tests_dir)
        )
        self.base_branch = str(base_branch)
        self.gh_runner = gh_runner

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def open_pr(
        self,
        proposal: FeatureProposal,
        backtest_result: BacktestResult,
        *,
        dry_run: bool = True,
    ) -> Optional[PROpenResult]:
        """Materialize the proposal and (optionally) open a PR.

        Refuses to do ANYTHING for proposals that didn't pass the gate;
        returns ``None`` and logs a warning.
        """
        if not backtest_result.passed_gate:
            LOGGER.warning(
                "PROpener: refusing to open PR for %r — passed_gate=False",
                proposal.name,
            )
            return None

        slug = _slugify(proposal.name)
        feature_path = self.features_dir / f"{slug}.py"
        test_path = self.tests_dir / f"test_llm_proposed_{slug}.py"

        # Write files (skeleton does this in BOTH dry and live mode so
        # operators can audit the artifacts before any branch exists).
        self._write_feature_file(feature_path, proposal, backtest_result)
        self._write_test_stub(test_path, slug, proposal)

        branch_name = f"e3/llm-proposed-{slug}"
        pr_title = f"WIP: LLM-proposed feature {proposal.name}"
        pr_body = self._build_pr_body(proposal, backtest_result, slug=slug)
        gh_command = [
            "gh", "pr", "create",
            "--base", self.base_branch,
            "--head", branch_name,
            "--draft",
            "--title", pr_title,
            "--body", pr_body,
        ]

        if dry_run:
            LOGGER.info(
                "PROpener: dry-run; wrote %s + %s; would invoke %s",
                feature_path,
                test_path,
                gh_command,
            )
            return PROpenResult(
                feature_path=feature_path,
                test_path=test_path,
                branch_name=None,
                pr_url=None,
                gh_command=gh_command,
                dry_run=True,
            )

        if self.gh_runner is None:
            LOGGER.warning(
                "PROpener: live mode requested but no gh_runner wired; "
                "falling back to dry-run behavior"
            )
            return PROpenResult(
                feature_path=feature_path,
                test_path=test_path,
                branch_name=None,
                pr_url=None,
                gh_command=gh_command,
                dry_run=True,
            )

        try:
            result = self.gh_runner(gh_command)
        except Exception as exc:  # noqa: BLE001 - never raise to the WeeklyJob loop
            LOGGER.error("PROpener: gh_runner crashed (%r)", exc)
            result = {}

        pr_url = None
        if isinstance(result, dict):
            pr_url = result.get("url") or result.get("pr_url")
        return PROpenResult(
            feature_path=feature_path,
            test_path=test_path,
            branch_name=branch_name,
            pr_url=pr_url,
            gh_command=gh_command,
            dry_run=False,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _write_feature_file(
        self,
        path: Path,
        proposal: FeatureProposal,
        backtest_result: BacktestResult,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f'"""LLM-proposed feature: {proposal.name}\n\n'
            f"Description: {proposal.description}\n"
            f"Expected Sharpe lift (LLM): {proposal.expected_lift_sharpe:+.4f}\n"
            f"Stub backtest Sharpe lift: {backtest_result.sharpe_lift:+.4f} "
            f"(threshold {backtest_result.gate_threshold:+.4f})\n"
            f"Risk notes: {proposal.risk_notes}\n"
            f"Proposed at: {proposal.proposed_at_utc}\n\n"
            "SECURITY: This code was written by an LLM. Operator review is REQUIRED\n"
            "before merging. Do NOT import from this module in production code\n"
            'until a human has reviewed every line.\n"""\n'
        )
        # Include the proposal source at the bottom.
        body = "\n\n# --- LLM-proposed implementation ---\n" + proposal.python_code + "\n"
        path.write_text(header + body, encoding="utf-8")

    def _write_test_stub(self, path: Path, slug: str, proposal: FeatureProposal) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = (
            f'"""Stub test for LLM-proposed feature {proposal.name!r}.\n\n'
            "The skeleton intentionally does NOT exercise the LLM-proposed\n"
            "code automatically — running untrusted code as part of the\n"
            "regular test suite is unsafe. Once a human reviewer signs off,\n"
            "they can replace ``self.skipTest(...)`` with real assertions.\n"
            '"""\n'
            "from __future__ import annotations\n\n"
            "import unittest\n\n\n"
            f"class LLMProposed{slug.title().replace('_', '')}Tests(unittest.TestCase):\n"
            f"    def test_review_required(self):\n"
            f"        self.skipTest(\n"
            f"            \"LLM-proposed feature {proposal.name!r}; awaiting human review.\"\n"
            f"        )\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n"
        )
        path.write_text(contents, encoding="utf-8")

    def _build_pr_body(
        self,
        proposal: FeatureProposal,
        backtest_result: BacktestResult,
        *,
        slug: str,
    ) -> str:
        return (
            f"## LLM-proposed feature: {proposal.name}\n\n"
            f"**Description**: {proposal.description}\n\n"
            f"**Stub backtest Sharpe lift**: {backtest_result.sharpe_lift:+.4f} "
            f"(threshold {backtest_result.gate_threshold:+.4f})\n\n"
            f"**LLM expected Sharpe lift**: {proposal.expected_lift_sharpe:+.4f}\n\n"
            f"**Risk notes**: {proposal.risk_notes}\n\n"
            "## Caveats — READ BEFORE MERGING\n"
            "- This code was written by an LLM. Every line needs human review.\n"
            "- The Sharpe lift was computed with a stub backtest, not real "
            "retraining. Do not trust it.\n"
            "- The stub safety check rejects only the most obvious unsafe "
            "constructs; subtle attacks may have slipped through.\n"
            "- The stub test file calls ``self.skipTest`` — replace it with "
            "real assertions before merging.\n\n"
            f"Generated by the E3 LLM Strategy-Gen Loop (slug: ``{slug}``).\n"
        )


__all__ = [
    "PROpener",
    "PROpenResult",
    "DEFAULT_FEATURES_DIR",
    "DEFAULT_TESTS_DIR",
    "DEFAULT_BASE_BRANCH",
]
