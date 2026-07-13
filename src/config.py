from __future__ import annotations

from pathlib import Path
from typing import Optional, ClassVar

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
    _USE_PYDANTIC_SETTINGS = True
except Exception:
    # Fallback: pydantic-settings not installed (or older pydantic stack).
    try:
        from pydantic.v1 import BaseSettings, Field  # type: ignore
    except Exception:
        from pydantic import BaseSettings, Field  # type: ignore
    SettingsConfigDict = None
    _USE_PYDANTIC_SETTINGS = False

# NOTE: If you need CLI overrides, prefer environment variables or set fields directly
# before importing `cfg`, e.g. `TRADING_CAPITAL=20000 python src/backtest.py`.

FEATURE_VERSION = "v2026-02-119"
# Polymarket charges a maker/taker fee on each trade (2% as of 2026-Q2).
# Used by ``risk_management_agent.risk_engine`` to discount the calibrated
# edge before it feeds Kelly sizing. Expressed in basis points.
POLYMARKET_FEE_BPS: int = 200
_CONFIG_DIR = Path(__file__).resolve().parent
_ENV_FILES = (
    str(_CONFIG_DIR.parent / ".env"),
    str(_CONFIG_DIR / ".env"),
)


class TradingConfig(BaseSettings):
    FEATURE_VERSION: ClassVar[str] = FEATURE_VERSION
    # Data / model
    data_dir: str = Field("eth_1m_data", env="DATA_DIR")
    model_dir: str = Field("model_sanity", env="MODEL_DIR")
    window_size: Optional[int] = Field(None, env="WINDOW_SIZE")
    batch_size: int = Field(512, env="BATCH_SIZE")
    chunksize: int = Field(300_000, env="CHUNKSIZE")
    device: str = Field("auto", env="DEVICE")

    # Portfolio / execution
    capital: float = Field(10_000.0, env="CAPITAL")
    cooldown: int = Field(3, env="COOLDOWN")
    slippage_pct: float = Field(0.0002, env="SLIPPAGE_PCT")
    fee_pct: float = Field(0.0008, env="FEE_PCT")
    use_atr_stops: bool = Field(True, env="USE_ATR_STOPS")
    atr_tp_mult: float = Field(1.8, env="ATR_TP_MULT")
    atr_sl_mult: float = Field(1.0, env="ATR_SL_MULT")
    tp_pct: float = Field(0.005, env="TP_PCT")
    sl_pct: float = Field(0.0025, env="SL_PCT")
    use_regime_filter: bool = Field(True, env="USE_REGIME_FILTER")
    min_atr_pct: float = Field(0.001, env="MIN_ATR_PCT")

    # Classification gating
    thr_long: float = Field(0.75, env="THR_LONG")
    thr_short: float = Field(0.75, env="THR_SHORT")
    margin: float = Field(0.25, env="MARGIN")
    consensus: int = Field(2, env="CONSENSUS")
    use_hard_gating: bool = Field(True, env="USE_HARD_GATING")

    # Profit mode
    profit_mode: bool = Field(True, env="PROFIT_MODE")

    # Regression thresholds
    up_thr: float = Field(0.002, env="UP_THR")
    down_thr: float = Field(0.002, env="DOWN_THR")

    # Live trading
    symbol: str = Field("ETH/USDT", env="SYMBOL")
    interval_sec: float = Field(5.0, env="INTERVAL_SEC")
    starting_cash: float = Field(10_000.0, env="STARTING_CASH")
    allow_shorts: bool = Field(False, env="ALLOW_SHORTS")
    max_leverage: float = Field(1.0, env="MAX_LEVERAGE")
    # Halal (Shariah-compliant) trading mode. When True, the live supervisor
    # hard-enforces LONG-ONLY + SPOT-ONLY execution and fail-closes on any
    # non-compliant order: no shorts (selling an asset you don't own), no
    # perpetual-futures / leverage / funding-rate venues (riba + gharar). The
    # Polymarket prediction-market stack (maisir/gambling) is scan-and-advise
    # only and is never routed to the trading engine. Default ON — this is the
    # operational default for every CLI run of the bot. Disable per-run with
    # --no-halal-mode, or globally with HALAL_MODE=0 in the environment.
    HALAL_MODE: bool = Field(True, env="HALAL_MODE")
    risk_pct_per_trade: float = Field(0.004, env="RISK_PCT_PER_TRADE")
    csv_path: str = Field("", env="CSV_PATH")
    testnet: bool = Field(True, env="TESTNET")
    real: bool = Field(False, env="REAL")

    # AWS / deployment
    aws_region: str = Field("us-east-1", env="AWS_REGION")
    sagemaker_role_arn: Optional[str] = Field(None, env="SAGEMAKER_ROLE_ARN")
    model_s3_path: Optional[str] = Field(None, env="MODEL_S3_PATH")
    endpoint_name: str = Field("eth-endpoint", env="ENDPOINT_NAME")
    endpoint_instance_type: str = Field("ml.m5.large", env="ENDPOINT_INSTANCE_TYPE")
    endpoint_instances: int = Field(1, env="ENDPOINT_INSTANCES")
    serverless_memory_mb: int = Field(2048, env="SERVERLESS_MEMORY_MB")
    serverless_max_concurrency: int = Field(10, env="SERVERLESS_MAX_CONCURRENCY")

    # CoinDesk WS
    coindesk_ws_url: str = Field("wss://data-api.coindesk.com/streaming/spot", env="COINDESK_WS_URL")
    coindesk_market: str = Field("coinbase", env="COINDESK_WS_MARKET")
    coindesk_instrument: str = Field("ETH-USDT", env="COINDESK_WS_INSTRUMENT")
    coindesk_api_key: Optional[str] = Field(None, env="COINDESK_API_KEY")
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", env="GEMINI_MODEL")
    gemini_timeout_s: int = Field(30, env="GEMINI_TIMEOUT_S")
    gemini_use_search_grounding: bool = Field(True, env="GEMINI_USE_SEARCH_GROUNDING")

    # Exit policy (Sprint 1 Wave 1B + Wave 2). All knobs are independently
    # togglable — the float / int thresholds use None to disable, the
    # master switch uses False. ``EXIT_POLICY_ENABLED`` flipped to True in
    # Wave 2 once the high-water-mark plumbing and reason-tagged force-flat
    # paths landed; operators can still opt out with
    # ``EXIT_POLICY_ENABLED=0`` to reproduce the legacy "no exit policy"
    # behavior (regression sentinel).
    STOP_LOSS_PCT: Optional[float] = Field(-0.004, env="STOP_LOSS_PCT")
    TAKE_PROFIT_PCT: Optional[float] = Field(0.008, env="TAKE_PROFIT_PCT")
    TIME_STOP_BARS: Optional[int] = Field(20, env="TIME_STOP_BARS")
    TRAILING_STOP_PCT: Optional[float] = Field(None, env="TRAILING_STOP_PCT")
    EXIT_SIGNAL_REVERSAL: bool = Field(False, env="EXIT_SIGNAL_REVERSAL")
    EXIT_POLICY_ENABLED: bool = Field(True, env="EXIT_POLICY_ENABLED")

    # Kelly sizing (Sprint 1 Wave 2). When ``KELLY_SIZING_ENABLED`` is True
    # and the wired predictor exposes a non-None ``_last_resolved_kelly_pct``
    # (set by ``XGBoostPredictor`` from the regime-memory lookup), the
    # supervisor sizes new entries as ``bankroll * resolved_pct`` clipped
    # to ``[KELLY_FLOOR_PCT, KELLY_CAP_PCT]``. When the predictor doesn't
    # surface a value, OR the master switch is False, sizing falls back to
    # the legacy ``bankroll * risk_pct_per_trade`` path (preserving every
    # existing test fixture). Defaults are deliberately conservative:
    # 0.5% floor (any opened trade must be at least meaningful) and 5% cap
    # (Kelly half-fraction ceiling — protects against blown-up resolved
    # values from a low-confidence regime match).
    KELLY_SIZING_ENABLED: bool = Field(True, env="KELLY_SIZING_ENABLED")
    KELLY_FLOOR_PCT: float = Field(0.005, env="KELLY_FLOOR_PCT")
    KELLY_CAP_PCT: float = Field(0.05, env="KELLY_CAP_PCT")

    if _USE_PYDANTIC_SETTINGS and SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=_ENV_FILES,
            env_file_encoding="utf-8",
            extra="ignore",
        )
    else:
        class Config:
            env_file = _ENV_FILES
            env_file_encoding = "utf-8"
            extra = "ignore"


def load_config() -> TradingConfig:
    return TradingConfig()


# Singleton for convenience
cfg = load_config()
