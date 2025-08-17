#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launch_sagemaker_job.py

Fixed layout:
- Trainer entry_point must be:  ./aws_train_model.py
- Local training data directory: eth_1m_data/   (uploaded to S3, unless you pass an s3:// URI)

Reads defaults from .env; CLI overrides .env.

Minimal run:
  python launch_sagemaker_job.py

Common override:
  python launch_sagemaker_job.py --instance-type ml.g5.2xlarge --epochs 10 --batch-size 512 --seq-len 150
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Uploader

# ---- .env support ----
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_, **__):
        return False

load_dotenv()

# ---------- env helpers ----------
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v not in (None, "") else default

def _env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    v = _env(key)
    if v is None: return default
    try: return int(v)
    except Exception: return default

def _env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    v = _env(key)
    if v is None: return default
    try: return float(v)
    except Exception: return default

def _env_bool_or_int(key: str, default: int = 0) -> int:
    v = _env(key)
    if v is None: return int(default)
    vl = v.strip().lower()
    if vl in ("1","true","yes","y","on"): return 1
    if vl in ("0","false","no","n","off"): return 0
    try: return int(v)
    except Exception: return int(default)

# ---------- argparse ----------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch SageMaker training job for ./aws_train_model.py")

    # AWS / infra
    p.add_argument("--region", type=str,
                   default=_env("AWS_REGION", _env("AWS_DEFAULT_REGION")),
                   help="AWS region.")
    p.add_argument("--role", type=str,
                   default=_env("SAGEMAKER_ROLE_ARN", _env("SM_ROLE_ARN", _env("IAM_ROLE_NAME"))),
                   help="SageMaker role ARN or role NAME.")
    p.add_argument("--bucket", type=str,
                   default=_env("S3_BUCKET", _env("S3_BUCKET_NAME")),
                   help="S3 bucket (defaults to SDK session bucket).")
    p.add_argument("--prefix", type=str,
                   default=_env("S3_PREFIX", "autopilot-lstm"),
                   help="S3 prefix for artifacts/uploads.")
    p.add_argument("--instance-type", type=str,
                   default=_env("INSTANCE_TYPE", _env("TRAIN_INSTANCE_TYPE", "ml.c5.2xlarge")),
                   help="Training instance type.")
    p.add_argument("--instance-count", type=int,
                   default=_env_int("INSTANCE_COUNT", 1),
                   help="Number of instances.")
    p.add_argument("--image-uri", type=str, default=_env("IMAGE_URI"),
                   help="(Optional) custom training image URI.")
    p.add_argument("--job-name", type=str, default=_env("JOB_NAME"),
                   help="(Optional) explicit job name.")

    # Data (default fixed to eth_1m_data/)
    p.add_argument("--data", type=str,
                   default=_env("DATA_PATH", _env("DATA_DIR", "eth_1m_data")),
                   help="Training data: local dir (uploaded) or s3:// URI (default: ./eth_1m_data)")

    # Trainer hyperparams (the trainer also accepts legacy aliases)
    p.add_argument("--epochs", type=int, default=_env_int("EPOCHS", 10))
    p.add_argument("--batch-size", type=int, default=_env_int("BATCH_SIZE", 128))
    p.add_argument("--hidden-size", type=int, default=_env_int("HIDDEN_SIZE", 128))
    p.add_argument("--num-layers", type=int, default=_env_int("NUM_LAYERS", 2))
    p.add_argument("--dropout", type=float, default=_env_float("DROPOUT", 0.1))
    p.add_argument("--bidirectional", type=int, default=_env_bool_or_int("BIDIRECTIONAL", 0))
    p.add_argument("--num-classes", type=int, default=_env_int("NUM_CLASSES", 3))
    p.add_argument("--seq-len", type=int, default=_env_int("SEQ_LEN", 60))
    p.add_argument("--val-frac", type=float, default=_env_float("VAL_FRAC", 0.2))
    p.add_argument("--val-months", type=int, default=_env_int("VAL_MONTHS"))
    p.add_argument("--learning-rate", type=float, default=_env_float("LEARNING_RATE", 1e-3))
    p.add_argument("--weight-decay", type=float, default=_env_float("WEIGHT_DECAY", 0.0))
    p.add_argument("--scale-features", type=int, default=_env_bool_or_int("SCALE_FEATURES", 1))
    p.add_argument("--use-class-weights", type=int, default=_env_bool_or_int("USE_CLASS_WEIGHTS", 1))
    p.add_argument("--accumulate", type=int, default=_env_int("ACCUMULATE", 1),
                   help="Gradient accumulation steps.")

    # Env / runtime
    p.add_argument("--chunk-size", type=int, default=_env_int("CHUNK_SIZE", 200000),
                   help="Rows per chunk for trainer; exported as env CHUNK_SIZE.")
    return p

# ---------- region, role, s3 ----------
def _resolve_region(region_opt: Optional[str]) -> str:
    if region_opt: return region_opt
    session = boto3.session.Session()
    if session.region_name: return session.region_name
    raise RuntimeError("No AWS region set. Use --region or AWS_REGION/AWS_DEFAULT_REGION.")

def _resolve_role_arn(role_or_arn: str, region: str) -> str:
    if role_or_arn.startswith("arn:aws:iam::"):
        return role_or_arn
    iam = boto3.client("iam", region_name=region)
    try:
        resp = iam.get_role(RoleName=role_or_arn)
        return resp["Role"]["Arn"]
    except Exception as e:
        raise SystemExit(
            f"Could not resolve role name '{role_or_arn}' to an ARN. "
            f"Set SAGEMAKER_ROLE_ARN in .env or pass --role. Underlying error: {e}"
        )

def _prepare_s3_data_uri(sess: sagemaker.Session, bucket: Optional[str], prefix: str, data_arg: str) -> str:
    # If you pass an s3:// URI, we use it directly; otherwise upload local eth_1m_data/
    if data_arg.startswith("s3://"):
        return data_arg
    local_path = Path(data_arg).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Data path not found: {local_path}")
    dest_bucket = bucket or sess.default_bucket()
    dest_prefix = f"{prefix}/eth_1m_data"
    s3_uri = f"s3://{dest_bucket}/{dest_prefix}"
    print(f"[INFO] Uploading {local_path} -> {s3_uri}")
    uploaded = S3Uploader.upload(str(local_path), s3_uri)
    return uploaded

# ---------- hyperparams ----------
def _hyperparams_from_args(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "data": "/opt/ml/input/data/train",
        "epochs": str(args.epochs),
        "batch-size": str(args.batch_size),
        "hidden-size": str(args.hidden_size),
        "num-layers": str(args.num_layers),
        "dropout": str(args.dropout),
        "bidirectional": str(int(args.bidirectional)),
        "num-classes": str(args.num_classes),
        "seq-len": str(args.seq_len),
        "learning-rate": str(args.learning_rate),
        "weight-decay": str(args.weight_decay),
        "scale-features": str(int(args.scale_features)),
        "use-class-weights": str(int(args.use_class_weights)),
        "accumulate": str(int(args.accumulate)),
        **({"val_months": str(int(args.val_months))} if args.val_months is not None else {"val-frac": str(args.val_frac)}),
    }

# ---------- main ----------
def main():
    args = build_arg_parser().parse_args()

    # Fixed trainer location: ./aws_train_model.py
    repo_root = Path(__file__).resolve().parent
    entry_point_rel = "./aws_train_model.py"
    trainer_path = repo_root / entry_point_rel
    if not trainer_path.exists():
        raise SystemExit(f"Trainer not found at {trainer_path}. Expected path is './aws_train_model.py'.")

    if not args.role:
        raise SystemExit("Missing SageMaker role. Set SAGEMAKER_ROLE_ARN (or IAM_ROLE_NAME) in .env or pass --role.")

    region = _resolve_region(args.region)
    role_arn = _resolve_role_arn(args.role, region)
    print(f"[INFO] Using region: {region}")
    print(f"[INFO] Using role:   {role_arn}")
    print(f"[INFO] Using source_dir: {repo_root}")
    print(f"[INFO] Using entry_point: {entry_point_rel}")

    sm_sess = sagemaker.Session(boto3.session.Session(region_name=region))
    bucket = args.bucket or sm_sess.default_bucket()
    print(f"[INFO] Using bucket: {bucket}")
    print(f"[INFO] Using prefix: {args.prefix}")

    # Data: default to local eth_1m_data/ (or s3:// if you pass it)
    s3_data_uri = _prepare_s3_data_uri(sm_sess, bucket, args.prefix, args.data)
    inputs = {"train": s3_data_uri}
    print(f"[INFO] Training data channel 'train' -> {s3_data_uri}")

    est_kwargs = dict(
        entry_point=entry_point_rel,     # relative to source_dir
        source_dir=str(repo_root),       # package project root
        role=role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        hyperparameters=_hyperparams_from_args(args),
        environment={"CHUNK_SIZE": str(int(args.chunk_size))},
        framework_version="2.1",
        py_version="py310",
        disable_profiler=True,
        debugger_hook_config=False,
        sagemaker_session=sm_sess,
    )
    if args.image_uri:
        est_kwargs["image_uri"] = args.image_uri

    estimator = PyTorch(**est_kwargs)
    job_name = args.job_name or sagemaker.utils.name_from_base(f"{args.prefix.replace('/', '-')}-train")
    print(f"[INFO] Launching job: {job_name}")
    estimator.fit(inputs=inputs, job_name=job_name)
    print("[DONE] Job submitted. Check SageMaker console for status.")

if __name__ == "__main__":
    main()