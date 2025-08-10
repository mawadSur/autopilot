import os
from pathlib import Path
from typing import Iterable

# 1) Load .env reliably (script dir first, then workspace root)
try:
    from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv
    SCRIPT_DIR = Path(__file__).resolve().parent
    local_env = SCRIPT_DIR / ".env"
    if local_env.exists():
        load_dotenv(local_env, override=False)
        print(f"[dotenv] Loaded {local_env}")
    else:
        # fall back to best match from current working dir upward
        found = find_dotenv(filename=".env", usecwd=True)
        if found:
            load_dotenv(found, override=False)
            print(f"[dotenv] Loaded {found}")
        else:
            print("[dotenv] No .env found (proceeding with process env)")
except Exception as e:
    print(f"[dotenv] Skipped loading .env: {e}")

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter
from botocore.exceptions import ClientError


def require_env(keys: Iterable[str]) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        where = (Path(__file__).resolve().parent / ".env")
        raise RuntimeError(
            "Missing required env vars: "
            + ", ".join(missing)
            + f"\nTip: create/update {where} (or a project-root .env) with those keys, "
              "or export them in your shell before running."
        )

def upload_to_s3(local_folder: str, s3_bucket: str, s3_prefix: str) -> str:
    s3 = boto3.client('s3')
    for dirpath, _, filenames in os.walk(local_folder):
        for f in filenames:
            local_path = os.path.join(dirpath, f)
            rel = os.path.relpath(local_path, local_folder)
            s3_key = f"{s3_prefix}/{rel}"
            try:
                s3.head_object(Bucket=s3_bucket, Key=s3_key)
                print(f"[SKIP] s3://{s3_bucket}/{s3_key} already exists")
                continue
            except ClientError as e:
                if e.response.get('Error', {}).get('Code') != '404':
                    raise
            print(f"[UPLOAD] {local_path} -> s3://{s3_bucket}/{s3_key}")
            s3.upload_file(local_path, s3_bucket, s3_key)
    return f"s3://{s3_bucket}/{s3_prefix}"


# --------- Read & validate env ---------
require_env(["S3_BUCKET_NAME", "SAGEMAKER_ROLE_ARN"])
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
IAM_ROLE_ARN   = os.getenv("SAGEMAKER_ROLE_ARN")

DATA_DIR       = os.getenv("DATA_DIR", "eth_1m_data")
S3_PREFIX      = os.getenv("S3_PREFIX", "eth-price-prediction")
INSTANCE_TYPE  = os.getenv("TRAIN_INSTANCE_TYPE", "ml.g4dn.xlarge")
REGION         = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ENDPOINT_INSTANCE_TYPE = os.getenv("ENDPOINT_INSTANCE_TYPE", "ml.g4dn.xlarge")
TUNER_MAX_JOBS       = int(os.getenv("TUNER_MAX_JOBS", "10"))
TUNER_MAX_PARALLEL   = int(os.getenv("TUNER_MAX_PARALLEL", "2"))

# Ensure Boto/SageMaker use the same region
boto3.setup_default_session(region_name=REGION)
sess = sagemaker.Session()

s3_data_path = upload_to_s3(DATA_DIR, S3_BUCKET_NAME, S3_PREFIX)
print(f"Data uploaded to: {s3_data_path}")

estimator = PyTorch(
    entry_point='aws_train_model.py',
    source_dir='src',
    role=IAM_ROLE_ARN,
    framework_version='2.0.0',
    py_version='py310',
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    hyperparameters={}
)
metric_defs = [{'Name': 'val:loss', 'Regex': r'val ([+-]?\d*\.\d+|\d+)'}]

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name='val:loss',
    hyperparameter_ranges={
        'hidden_size': ContinuousParameter(32, 256),
        'dropout': ContinuousParameter(0.0, 0.6),
    },
    metric_definitions=metric_defs,
    max_jobs=TUNER_MAX_JOBS,
    max_parallel_jobs=TUNER_MAX_PARALLEL,
    objective_type='Minimize',
)

print("Starting SageMaker Hyperparameter Tuning job...")
tuner.fit({'train': s3_data_path})
tuner.wait()
print("✅ Hyperparameter Tuning job finished.")

print("Deploying the best model...")
predictor = tuner.deploy(
    initial_instance_count=1,
    instance_type=ENDPOINT_INSTANCE_TYPE,
    entry_point='inference.py'
)
print(f"✅ Best model deployed. Endpoint: {predictor.endpoint_name}")
