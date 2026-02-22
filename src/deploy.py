#!/usr/bin/env python3
"""
Unified deployment/processing/training launcher for SageMaker.

Modes:
  - processing : run a Processing job with an entry script
  - serverless : deploy a PyTorch model to a serverless endpoint
  - training   : launch a PyTorch training job
"""

from __future__ import annotations

import argparse
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import boto3
import pandas as pd
import sagemaker
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.pytorch import PyTorchModel, PyTorch
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.inputs import TrainingInput

from config import cfg
from utils import normalize_headers

load_dotenv()


def _required(val: Optional[str], name: str) -> str:
    if not val:
        raise ValueError(f"Missing required {name}")
    return val


def str2bool(v) -> bool:
    return str(v).lower() in ("1", "true", "t", "yes", "y")


_RAW_SCHEMA_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "best_bid", "best_ask", "bid_size_l1", "ask_size_l1",
    "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10", "bid_depth_20", "ask_depth_20",
    "vwap_bid_5", "vwap_ask_5", "vwap_bid_10", "vwap_ask_10", "vwap_bid_20", "vwap_ask_20",
    "trade_count", "buy_count", "sell_count",
    "taker_buy_volume_base", "taker_sell_volume_base",
    "taker_buy_volume_quote", "taker_sell_volume_quote",
    "volume_quote",
]


def validate_raw_schema(files: Sequence[Union[str, Path]], sample_rows: int = 2000) -> Dict[str, List[str]]:
    """Validate sampled raw CSV schema and report missing raw columns per file."""
    missing_by_file: Dict[str, List[str]] = {}
    for fp in files:
        path = Path(fp)
        try:
            sample = pd.read_csv(path, nrows=sample_rows)
        except Exception as e:
            print(f"[schema] ERROR reading {path}: {e}")
            missing_by_file[str(path)] = list(_RAW_SCHEMA_COLUMNS)
            continue
        sample = normalize_headers(sample)
        missing = [c for c in _RAW_SCHEMA_COLUMNS if c not in sample.columns]
        if missing:
            print(f"[schema] {path} missing {len(missing)} raw cols: {missing}")
            missing_by_file[str(path)] = missing

    if not missing_by_file:
        print("[schema] All sampled files include required raw columns.")
    return missing_by_file


def _drop_features_for_missing_raw(
    desired_features: List[str],
    missing_raw_anywhere: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Drop features whose raw dependencies are missing in sampled files."""
    missing_set = set(missing_raw_anywhere)
    if not missing_set:
        return list(desired_features), []

    deps: Dict[str, set[str]] = {
        "best_bid": {"best_bid"},
        "best_ask": {"best_ask"},
        "bid_size_l1": {"bid_size_l1"},
        "ask_size_l1": {"ask_size_l1"},
        "bid_depth_5": {"bid_depth_5"},
        "ask_depth_5": {"ask_depth_5"},
        "bid_depth_10": {"bid_depth_10"},
        "ask_depth_10": {"ask_depth_10"},
        "bid_depth_20": {"bid_depth_20"},
        "ask_depth_20": {"ask_depth_20"},
        "vwap_bid_5": {"vwap_bid_5"},
        "vwap_ask_5": {"vwap_ask_5"},
        "vwap_bid_10": {"vwap_bid_10"},
        "vwap_ask_10": {"vwap_ask_10"},
        "vwap_bid_20": {"vwap_bid_20"},
        "vwap_ask_20": {"vwap_ask_20"},
        "trade_count": {"trade_count"},
        "buy_count": {"buy_count"},
        "sell_count": {"sell_count"},
        "taker_buy_volume_base": {"taker_buy_volume_base"},
        "taker_sell_volume_base": {"taker_sell_volume_base"},
        "taker_buy_volume_quote": {"taker_buy_volume_quote"},
        "taker_sell_volume_quote": {"taker_sell_volume_quote"},
        "volume_quote": {"volume_quote"},
        "mid": {"best_bid", "best_ask"},
        "spread_abs": {"best_bid", "best_ask"},
        "spread_pct": {"best_bid", "best_ask"},
        "microprice": {"best_bid", "best_ask", "bid_size_l1", "ask_size_l1"},
        "l1_imbalance": {"bid_size_l1", "ask_size_l1"},
        "mid_log_ret": {"best_bid", "best_ask"},
        "spread_z_60": {"best_bid", "best_ask"},
        "l2_imbalance_5": {"bid_depth_5", "ask_depth_5"},
        "l2_imbalance_10": {"bid_depth_10", "ask_depth_10"},
        "l2_imbalance_20": {"bid_depth_20", "ask_depth_20"},
        "depth_ratio_5": {"bid_depth_5", "ask_depth_5"},
        "depth_ratio_10": {"bid_depth_10", "ask_depth_10"},
        "depth_ratio_20": {"bid_depth_20", "ask_depth_20"},
        "book_pressure_5": {"bid_depth_5", "ask_depth_5"},
        "total_taker_vol_base": {"taker_buy_volume_base", "taker_sell_volume_base"},
        "ofi_base": {"taker_buy_volume_base", "taker_sell_volume_base"},
        "ofi_ratio": {"taker_buy_volume_base", "taker_sell_volume_base"},
        "buy_sell_count_imb": {"buy_count", "sell_count"},
        "avg_trade_size_base": {"trade_count", "volume"},
        "avg_trade_size_quote": {"trade_count", "volume_quote"},
        "ofi_over_depth_10": {"taker_buy_volume_base", "taker_sell_volume_base", "bid_depth_10", "ask_depth_10"},
        "spread_times_imbalance": {"best_bid", "best_ask", "bid_size_l1", "ask_size_l1"},
    }

    dropped = [f for f in desired_features if f in deps and not deps[f].isdisjoint(missing_set)]
    filtered = [f for f in desired_features if f not in dropped]
    return filtered, dropped


def _list_local_csvs(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    if path.suffix.lower() == ".csv" and path.exists():
        return [path]
    return []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified SageMaker deploy/processing/training launcher")
    p.add_argument("--mode", choices=["processing", "serverless", "training"], default="training")

    # shared
    p.add_argument("--region", default=cfg.aws_region)
    p.add_argument("--role-arn", default=os.getenv("SAGEMAKER_ROLE_ARN", cfg.sagemaker_role_arn))

    # processing
    p.add_argument("--entry-point", help="Processing entry script path")
    p.add_argument("--source-dir", default=".")
    p.add_argument("--data-s3-uri", default=None)
    p.add_argument("--model-s3-uri", default=None)
    p.add_argument("--output-bucket", default=os.getenv("S3_BUCKET_NAME"))
    p.add_argument("--output-prefix", default="processing-output")
    p.add_argument("--instance-type", default="ml.m5.48xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--volume-size", type=int, default=30)
    p.add_argument("--arguments", nargs="*", default=[])

    # serverless
    p.add_argument("--model-s3", default=cfg.model_s3_path)
    p.add_argument("--endpoint-name", default=cfg.endpoint_name)
    p.add_argument("--framework-version", default=os.getenv("FRAMEWORK_VERSION", "2.2"))
    p.add_argument("--py-version", default=os.getenv("PY_VERSION", "py311"))
    p.add_argument("--serverless-memory", type=int, default=cfg.serverless_memory_mb)
    p.add_argument("--serverless-max-concurrency", type=int, default=cfg.serverless_max_concurrency)

    # training
    p.add_argument("--train-s3-uri", default=None)
    p.add_argument("--local-train-dir", default=cfg.data_dir)
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET_NAME"))
    p.add_argument("--prefix", default=os.getenv("S3_PREFIX", "eth-training"))
    p.add_argument("--train-instance-type", default=os.getenv("TRAIN_INSTANCE_TYPE", "ml.g5.xlarge"))
    p.add_argument("--train-instance-count", type=int, default=int(os.getenv("TRAIN_INSTANCE_COUNT", 1)))
    p.add_argument("--train-job-name", default=None)
    # training hyperparameters (match launch_sagemaker_job.py)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--seq-lens", type=str, default=None)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    p.add_argument("--disable-scaling", type=str2bool, default=False)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--price-col", type=str, default="close")
    p.add_argument("--model-type", type=str, default="transformer")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--time-limit", type=int, default=5)
    p.add_argument("--fee-pct", type=float, default=0.0001)
    p.add_argument("--slippage-pct", type=float, default=0.0001)
    p.add_argument("--cost-mult", type=float, default=1.5)
    p.add_argument("--k-tp", type=float, default=1.2)
    p.add_argument("--k-sl", type=float, default=1.0)
    p.add_argument("--atr-col", type=str, default="atr_14")
    p.add_argument("--feature-cols", nargs="*", default=None)

    return p.parse_args()


def run_processing(args: argparse.Namespace) -> None:
    sess = boto3.session.Session(region_name=args.region)
    processor = PyTorchProcessor(
        framework_version=os.getenv("PYTORCH_VERSION", "2.2"),
        py_version=os.getenv("PYTHON_VERSION", "py311"),
        role=_required(args.role_arn, "role-arn"),
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size_in_gb=args.volume_size,
        sagemaker_session=None,
    )

    inputs = []
    if args.data_s3_uri:
        inputs.append(ProcessingInput(source=args.data_s3_uri, destination="/opt/ml/processing/input/data"))
    if args.model_s3_uri:
        inputs.append(ProcessingInput(source=args.model_s3_uri, destination="/opt/ml/processing/input/model"))

    bucket = _required(args.output_bucket, "output-bucket")
    prefix = args.output_prefix.rstrip("/")
    output_s3 = f"s3://{bucket}/{prefix}/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    outputs = [ProcessingOutput(source="/opt/ml/processing/output", destination=output_s3)]

    processor.run(
        code=_required(args.entry_point, "entry-point"),
        source_dir=str(Path(args.source_dir).resolve()),
        inputs=inputs,
        outputs=outputs,
        arguments=args.arguments,
        job_name=f"processing-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
    )
    print(f"Processing output: {output_s3}")


def run_serverless(args: argparse.Namespace) -> None:
    role = _required(args.role_arn, "role-arn")
    model_s3 = _required(args.model_s3, "model-s3")
    endpoint_name = _required(args.endpoint_name, "endpoint-name")

    sm_session = sagemaker.Session(boto3.Session(region_name=args.region))
    model = PyTorchModel(
        model_data=model_s3,
        role=role,
        entry_point="inference.py",
        source_dir=str(Path(__file__).resolve().parent),
        framework_version=args.framework_version,
        py_version=args.py_version,
        sagemaker_session=sm_session,
    )
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=args.serverless_memory,
        max_concurrency=args.serverless_max_concurrency,
    )
    predictor = model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=endpoint_name,
    )
    print(f"Serverless endpoint: {predictor.endpoint_name}")


def ensure_data_in_s3(local_dir: Path, bucket: str, prefix: str) -> str:
    """Check if training data already exists in S3. Upload only if missing."""
    s3 = boto3.client("s3")
    train_prefix = f"{prefix.rstrip('/')}/train/"
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=train_prefix, MaxKeys=1)
        if "Contents" in resp and len(resp["Contents"]) > 0:
            print(f"[data] Reusing existing S3 data: s3://{bucket}/{train_prefix}")
            return f"s3://{bucket}/{train_prefix}"
    except ClientError as e:
        raise RuntimeError(f"Failed to list S3 prefix {train_prefix}: {e}")

    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"Local training dir not found: {local_dir}")

    print(f"[data] Uploading {local_dir} → s3://{bucket}/{train_prefix} ...")
    uploaded = 0
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        key = f"{train_prefix}{rel}"
        extra = {}
        ctype, _ = mimetypes.guess_type(p.name)
        if ctype:
            extra["ContentType"] = ctype
        s3.upload_file(str(p), bucket, key, ExtraArgs=extra if extra else None)
        uploaded += 1
    print(f"[data] Uploaded {uploaded} files.")
    return f"s3://{bucket}/{train_prefix}"


def run_training(args: argparse.Namespace) -> None:
    role = _required(args.role_arn, "role-arn")
    local_dir = Path(args.local_train_dir).resolve()
    files = _list_local_csvs(local_dir)
    # Schema validation (prevents training with missing raw columns)
    print("[schema] Validating raw CSV schema...")
    missing = validate_raw_schema(files)  # reuse function from train_model.py or copy it
    if missing:
        print(f"[schema] WARNING: {len(missing)} files missing raw columns. Proceeding with feature dropping.")

    if args.feature_cols and missing:
        missing_raw_anywhere = sorted({col for cols in missing.values() for col in cols})
        filtered, dropped = _drop_features_for_missing_raw(list(args.feature_cols), missing_raw_anywhere)
        if dropped:
            print(f"[schema] Dropping {len(dropped)} feature-cols due to missing raw inputs: {dropped[:20]}")
        args.feature_cols = filtered

    bucket = _required(args.bucket, "bucket")
    prefix = args.prefix.rstrip("/")

    if args.train_s3_uri:
        train_s3 = args.train_s3_uri
    else:
        train_s3 = ensure_data_in_s3(local_dir, bucket, f"{prefix}/train")

    job_name = args.train_job_name or f"train-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    hyperparameters = {
        "window-size": args.window_size,
        "seq-lens": args.seq_lens,
        "hidden-size": args.hidden_size,
        "num-layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
        "disable-scaling": args.disable_scaling,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "learning-rate": args.learning_rate,
        "weight-decay": args.weight_decay,
        "val-frac": args.val_frac,
        "accumulate": args.accumulate,
        "seed": args.seed,
        "price-col": args.price_col,
        "model-type": args.model_type,
        "task": args.task,
        "time-limit": args.time_limit,
        "fee-pct": args.fee_pct,
        "slippage-pct": args.slippage_pct,
        "cost-mult": args.cost_mult,
        "k-tp": args.k_tp,
        "k-sl": args.k_sl,
        "atr-col": args.atr_col,
    }
    if args.feature_cols:
        hyperparameters["feature-cols"] = " ".join(args.feature_cols)
    # Avoid sending None/"None" values to argparse in train_model.py
    hyperparameters = {
        k: v for k, v in hyperparameters.items()
        if v is not None and not (isinstance(v, str) and v.strip().lower() == "none")
    }

    estimator = PyTorch(
        entry_point="train_model.py",
        source_dir=str(Path(__file__).resolve().parent),
        role=role,
        framework_version=os.getenv("FRAMEWORK_VERSION", "2.2"),
        py_version=os.getenv("PY_VERSION", "py311"),
        instance_type=args.train_instance_type,
        instance_count=args.train_instance_count,
        hyperparameters=hyperparameters,
    )
    estimator.fit(TrainingInput(train_s3), job_name=job_name)
    print(f"Training job submitted: {job_name}")


def main():
    args = parse_args()
    if args.mode == "processing":
        run_processing(args)
    elif args.mode == "serverless":
        run_serverless(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
