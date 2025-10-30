#!/usr/bin/env python3
"""
Launch a SageMaker Processing job to run LLM backtesting on AWS GPU instances.

This script submits the llm_backtest.py script to run on SageMaker GPU instances,
allowing for distributed LLM inference across powerful AWS hardware.

Usage:
    python launch_llm_sagemaker_job.py --bucket your-bucket --data-s3-uri s3://bucket/data/ --model-s3-uri s3://bucket/model/
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# Load environment variables from .env if present
load_dotenv()

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def _s3_uri(bucket: str, key_prefix: str) -> str:
    key_prefix = key_prefix.lstrip("/")
    return f"s3://{bucket}/{key_prefix}"

def ensure_data_in_s3(local_dir: Path, s3_client, bucket: str, prefix: str) -> str:
    """Upload local data to S3 if not already present."""
    data_prefix = f"{prefix.rstrip('/')}/data/"
    try:
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=data_prefix, MaxKeys=1)
        already_there = ("Contents" in resp and len(resp["Contents"]) > 0)
    except ClientError as e:
        raise RuntimeError(f"Failed to list s3://{bucket}/{data_prefix} - {e}")

    if already_there:
        print(f"Data already exists in s3://{bucket}/{data_prefix}")
        return _s3_uri(bucket, data_prefix)

    print(f"Uploading {local_dir} to s3://{bucket}/{data_prefix}")
    for local_file in local_dir.rglob("*"):
        print('local_file = ', local_file)
        if local_file.is_file():
            rel_path = local_file.relative_to(local_dir)
            s3_key = f"{data_prefix}{rel_path}"
            try:
                s3_client.upload_file(str(local_file), bucket, s3_key)
                print(f"  Uploaded: {local_file.name}")
            except Exception as e:
                print(f"  Failed to upload {local_file}: {e}")
                raise

    return _s3_uri(bucket, data_prefix)

def ensure_model_in_s3(local_dir: Path, s3_client, bucket: str, prefix: str) -> str:
    """Upload local model to S3 if not already present."""
    model_prefix = f"{prefix.rstrip('/')}/model/"
    try:
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=model_prefix, MaxKeys=1)
        already_there = ("Contents" in resp and len(resp["Contents"]) > 0)
    except ClientError as e:
        raise RuntimeError(f"Failed to list s3://{bucket}/{model_prefix} - {e}")

    if already_there:
        print(f"Model already exists in s3://{bucket}/{model_prefix}")
        return _s3_uri(bucket, model_prefix)

    print(f"Uploading {local_dir} to s3://{bucket}/{model_prefix}")
    for local_file in local_dir.rglob("*"):
        if local_file.is_file():
            rel_path = local_file.relative_to(local_dir)
            s3_key = f"{model_prefix}{rel_path}"
            try:
                s3_client.upload_file(str(local_file), bucket, s3_key)
            except Exception as e:
                print(f"  Failed to upload {local_file}: {e}")
                raise

    return _s3_uri(bucket, model_prefix)

def build_parser():
    p = argparse.ArgumentParser(description="Launch SageMaker Processing job for LLM backtesting")
    
    # AWS/SageMaker configuration
    p.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    p.add_argument("--role-arn", default=os.getenv("SAGEMAKER_ROLE_ARN"))
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET_NAME"), required=False)
    p.add_argument("--prefix", default="llm-backtest")
    p.add_argument("--job-name", default=None)
    
    # Instance configuration (GPU instances for LLM)
    p.add_argument("--instance-type", default="ml.g5.xlarge")
    p.add_argument("--instance-count", type=int, default=int(os.getenv("TRAIN_INSTANCE_COUNT", 1)))
    p.add_argument("--volume-size", type=int, default=30, help="EBS volume size in GB")
    
    # Container configuration
    p.add_argument("--framework-version", default=os.getenv("FRAMEWORK_VERSION", "2.1.0"))
    p.add_argument("--py-version", default=os.getenv("PY_VERSION", "py310"))
    
    # Data sources
    p.add_argument("--data-s3-uri", help="S3 URI for input data (CSV files)")
    p.add_argument("--local-data-dir", default="../eth_1m_data", help="Local data directory to upload if --data-s3-uri not provided")
    p.add_argument("--model-s3-uri", help="S3 URI for trained model")
    p.add_argument("--local-model-dir", help="Local model directory to upload if --model-s3-uri not provided (optional for LLM-only backtesting)")
    
    # LLM backtest parameters
    p.add_argument("--llm-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Hugging Face model ID")
    p.add_argument("--llm-batch-size", type=int, default=32, help="LLM batch size")
    p.add_argument("--llm-max-new-tokens", type=int, default=8, help="Max new tokens for LLM")
    p.add_argument("--capital", type=float, default=10000.0, help="Starting capital")
    p.add_argument("--thr-long", type=float, default=0.55)
    p.add_argument("--thr-short", type=float, default=0.55)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--consensus", type=int, default=2)
    p.add_argument("--tp-pct", type=float, default=0.002, help="Take profit percentage")
    p.add_argument("--sl-pct", type=float, default=0.005, help="Stop loss percentage")
    p.add_argument("--atr-tp-mult", type=float, default=5.0, help="ATR multiplier for take-profit (larger wins)")
    p.add_argument("--atr-sl-mult", type=float, default=1.0, help="ATR multiplier for stop-loss (tight risk)")
    p.add_argument("--cooldown", type=int, default=0, help="Bars to wait after an exit before re-entering (faster)")
    p.add_argument("--use-atr-stops", action="store_true", default=False, help="Use ATR multipliers for TP/SL (disabled by default)")
    p.add_argument("--slippage-pct", type=float, default=0.0002, help="Per-side slippage fraction for realism (0.02%)")
    p.add_argument("--leverage", type=float, default=1.0, help="Leverage factor (e.g. 2.0 for 2x)")
    p.add_argument("--dynamic-sizing", action="store_true", default=True, help="Enable position sizing by risk percent")
    p.add_argument("--use-regime-filter", action="store_true", help="Enable regime filter")
    p.add_argument("--switch-less-aggressive", action="store_true", default=True, help="Switch to less aggressive TP/SL logic")
    
    return p

def main():
    args = build_parser().parse_args()

    if not args.role_arn:
        raise ValueError("Missing --role-arn (or SAGEMAKER_ROLE_ARN in .env)")
    if not args.bucket:
        raise ValueError("Missing --bucket (or S3_BUCKET_NAME in .env)")

    script_dir = Path(__file__).resolve().parent
    entry_point_path = script_dir / "llm_backtest.py"
    if not entry_point_path.exists():
        raise ValueError(
            f'No file named "llm_backtest.py" was found in directory "{script_dir}". '
            "Ensure llm_backtest.py is alongside launch_llm_sagemaker_job.py."
        )

    s3 = boto3.client("s3", region_name=args.region)

    # Handle data S3 URI
    if args.data_s3_uri:
        data_s3_uri = args.data_s3_uri
        print(f"\\nUsing provided data S3 URI: {data_s3_uri}")
    else:
        local_data_dir = (Path(args.local_data_dir)
                         if Path(args.local_data_dir).is_absolute()
                         else (script_dir / args.local_data_dir).resolve())
        data_s3_uri = ensure_data_in_s3(local_data_dir, s3, args.bucket, f"{args.prefix}/data")

    # Handle model S3 URI (optional for LLM-only backtesting)
    model_s3_uri = None
    if args.model_s3_uri:
        model_s3_uri = args.model_s3_uri
        print(f"Using provided model S3 URI: {model_s3_uri}")
    elif args.local_model_dir:
        local_model_dir = (Path(args.local_model_dir)
                          if Path(args.local_model_dir).is_absolute()
                          else (script_dir / args.local_model_dir).resolve())
        model_s3_uri = ensure_model_in_s3(local_model_dir, s3, args.bucket, f"{args.prefix}/model")
    else:
        # Default to local scaler directory
        local_model_dir = (script_dir / "last_model").resolve()
        print(f"Uploading default model dir: {local_model_dir}")
        model_s3_uri = ensure_model_in_s3(local_model_dir, s3, args.bucket, f"{args.prefix}/model")

    job_name = args.job_name or f"llm-backtest-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    output_s3_uri = _s3_uri(args.bucket, f"{args.prefix}/output/{job_name}")

    print("\\n=== SageMaker LLM Backtest Job Configuration ===")
    print(f"Region:             {args.region}")
    print(f"Role ARN:           {args.role_arn}")
    print(f"Instance Type:      {args.instance_type}")
    print(f"Instance Count:     {args.instance_count}")
    print(f"Transformers Version: 4.28.1")
    print(f"PyTorch Version:    2.0.0")
    print(f"Python Version:     py310")
    print(f"Data S3 URI:        {data_s3_uri}")
    print(f"Model S3 URI:       {model_s3_uri if model_s3_uri else 'None (LLM-only mode)'}")
    print(f"Output S3 URI:      {output_s3_uri}")
    print(f"Job Name:           {job_name}")
    print(f"LLM Model:          {args.llm_model}")

    # Build arguments for the processing script
    script_args = [
        "--data-dir", "/opt/ml/processing/input/data/data",
        "--llm-model", args.llm_model,
        "--llm-batch-size", str(args.llm_batch_size),
        "--llm-max-new-tokens", str(args.llm_max_new_tokens),
        "--capital", str(args.capital),
        "--tp-pct", str(args.tp_pct),
        "--sl-pct", str(args.sl_pct),
        "--llm-device", "cuda",
    ]
    
    # Add model directory only if provided
    if model_s3_uri:
        script_args.extend(["--model-dir", "/opt/ml/processing/input/model"])
    
    if args.dynamic_sizing:
        script_args.append("--dynamic-sizing")
    if args.use_regime_filter:
        script_args.append("--use-regime-filter")
    if args.switch_less_aggressive:
        script_args.append("--switch-less-aggressive")

    print("\\n--- Processing Script Arguments ---")
    for i in range(0, len(script_args), 2):
        arg_name = script_args[i] if i < len(script_args) else ""
        arg_value = script_args[i+1] if i+1 < len(script_args) else ""
        print(f"{arg_name:20s}: {arg_value}")

    # Create HuggingFaceProcessor
    processor = HuggingFaceProcessor(
    transformers_version="4.46.1",  # Base—upgrades via requirements.txt
    pytorch_version="2.3.0",
    py_version="py311",
    role=args.role_arn,
    instance_type=args.instance_type,
    instance_count=1
)

    # Define inputs and outputs
    inputs = [
        ProcessingInput(
            source=data_s3_uri,
            destination="/opt/ml/processing/input/data",
            input_name="data"
        )
    ]
    
    # Add model input only if provided
    inputs.append(
        ProcessingInput(
            source=f"s3://{args.bucket}/{args.prefix}/model/",
            destination="/opt/ml/processing/input/model",
            input_name="model"
        )
    )

    outputs = [
        ProcessingOutput(
            output_name="results",
            source="/opt/ml/processing/output",
            destination=output_s3_uri
        )
    ]

    print("\\nStarting SageMaker processing job...")
    source_dir = "src"
    processor.run(
        code="llm_backtest.py",
        source_dir=str(source_dir),
        inputs=inputs,
        outputs=outputs,
        arguments=script_args,
        job_name=job_name,
        wait=True,
        logs=True
    )

    print(f"\\n=== Job Complete ===")
    print(f"Results available at: {output_s3_uri}")
    print(f"Job name: {job_name}")



if __name__ == "__main__":
    main()