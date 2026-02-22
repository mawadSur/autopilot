from __future__ import annotations

from typing import Optional

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


class TradingConfig(BaseSettings):
    # Data / model
    data_dir: str = Field("eth_1m_data", env="DATA_DIR")
    model_dir: str = Field("model", env="MODEL_DIR")
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

    # Regression thresholds
    up_thr: float = Field(0.002, env="UP_THR")
    down_thr: float = Field(0.002, env="DOWN_THR")

    # Live trading
    symbol: str = Field("ETH/USDT", env="SYMBOL")
    interval_sec: float = Field(5.0, env="INTERVAL_SEC")
    starting_cash: float = Field(10_000.0, env="STARTING_CASH")
    allow_shorts: bool = Field(False, env="ALLOW_SHORTS")
    max_leverage: float = Field(1.0, env="MAX_LEVERAGE")
    risk_pct_per_trade: float = Field(0.004, env="RISK_PCT_PER_TRADE")
    capital: Optional[float] = Field(None, env="CAPITAL")
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

    if _USE_PYDANTIC_SETTINGS and SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"


def load_config() -> TradingConfig:
    return TradingConfig()


# Singleton for convenience
cfg = load_config()
