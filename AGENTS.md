# Repository Guidelines

## Project Structure & Module Organization

Core prediction-market code lives in `src/`. Key entrypoints are `main.py` for the Polymarket scanner, `src/orchestrator.py` for multi-agent research and calibration, and `social-narrative-agent/main.py` for OpenAI-based social narrative analysis. Legacy trading services remain under `src/` (`main.py`, `dashboard_server.py`, `dashboard_app.py`, training/backtest scripts). Tests are split between `tests/`, `tests/prediction_market_scanner/`, and `social-narrative-agent/tests/`.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate`: create and activate the local environment.
- `pip install -r requirements.txt`: install the full Python stack.
- `./.venv/bin/python main.py --help`: inspect scanner CLI options.
- `./.venv/bin/python src/orchestrator.py --help`: inspect orchestrator options.
- `./.venv/bin/python social-narrative-agent/main.py --help`: inspect social agent options.
- `env PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests/prediction_market_scanner -p 'test_main.py'`: run prediction-market tests that import from `src`.
- `./.venv/bin/python -m unittest discover social-narrative-agent/tests`: run social agent tests.

## Coding Style & Naming Conventions

Use 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and descriptive filenames like `test_orchestrator.py`. Match the existing Python style: type hints where practical, small focused helpers, and direct CLI-oriented modules. No formatter or linter is configured in-repo, so keep changes consistent with surrounding code and avoid broad style-only edits.

## Testing Guidelines

This repo uses mostly `unittest`, with some `pytest` usage in `tests/test_core.py`. Name new tests `test_*.py` and keep test classes and methods explicit about the unit under test. For `src/` imports, set `PYTHONPATH=src`; for isolated CLIs, prefer targeted `unittest discover` runs before broad suites.

## Commit & Pull Request Guidelines

Recent history uses short, capitalized summaries such as `Added logic` and `updated backtest and model`. Keep commit subjects brief, imperative, and specific, for example `Add scanner category filter`. PRs should state the affected surface (`main.py`, `src/orchestrator.py`, dashboard, etc.), list commands/tests run, note any new env vars, and include screenshots only for dashboard or UI changes.

## Security & Configuration Tips

Store secrets in `.env`, not in code. Common keys include `GEMINI_API_KEY`, `OPENAI_API_KEY`, and Reddit credentials. Avoid committing generated data, model artifacts, or large local outputs unless they are intentionally versioned.
