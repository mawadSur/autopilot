"""Read-only observability dashboard for the whale-follow SHADOW loop.

This package is OBSERVABILITY ONLY. It reads the append-only PnL ledger
(:mod:`state.pnl_ledger`) via :func:`portfolio_reporter.build_report` and serves
a self-contained web page over stdlib ``http.server``. It places NO orders and
never mutates the ledger or any other file.
"""
