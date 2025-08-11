import os
from pathlib import Path
from typing import Iterable

# ---- Load .env reliably (script dir first, then working dir) ----
try:
    from dotenv import load_dotenv, find_dotenv
    HERE = Path(__file__).resolve().parent
    env_local = HERE / ".env"
    if env_local.exists():
        load_dotenv(env_local, override=False)
        print(f"[dotenv] Loaded {env_local}")
    else:
        found = find_dotenv(filename=".env", usecwd=True)
        if found:
            load_dotenv(found, override=False)
            print(f"[dotenv] Loaded {found}")
        else:
            print("[dotenv] No .env found; using process env")
except Exception as e:
    print(f"[dotenv] Skipped loading .env: {e}")

# ---- Networking hygiene (avoid proxy issues) ----
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","all_proxy"):
    os.environ.pop(k, None)
no_proxy = os.environ.get("NO_PROXY","")
for s in (".amazonaws.com","amazonaws.com","169.254.169.254","localhost","127.0.0.1"):
    if s not in no_proxy:
        no_proxy = (no_proxy + "," if no_proxy else "") + s
os.environ["NO_PROXY"] = os.environ["no_proxy"] = no_proxy

import boto3
from botocore.config import Config
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter, CategoricalParameter
from botocore.exceptions import ClientError


def require_env(keys: Iterable[str]) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


# ---- Required env ----
require_env(["S3_BUCKET_NAME", "SAGEMAKER_ROLE_ARN"])
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET = os.getenv("S3_BUCKET_NAME")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
DATA_DIR = os.getenv("DATA_DIR", "eth_1m_data")
S3_PREFIX = os.getenv("S3_PREFIX", "eth-price-prediction/data")
INSTANCE_TYPE = os.getenv("TRAIN_INSTANCE_TYPE", "ml.g4dn.xlarge")
ENDPOINT_INSTANCE_TYPE = os.getenv("ENDPOINT_INSTANCE_TYPE", "ml.g4dn.xlarge")
TUNER_MAX_JOBS = int(os.getenv("TUNER_MAX_JOBS", "2"))
TUNER_MAX_PARALLEL = int(os.getenv("TUNER_MAX_PARALLEL", "1"))

# ---- Sessions/clients with explicit region + resilient timeouts ----
boto_sess = boto3.Session(region_name=REGION)
sm_client = boto_sess.client("sagemaker", config=Config(retries={"max_attempts": 10, "mode": "standard"}))
s3_client = boto_sess.client("s3", config=Config(read_timeout=300, connect_timeout=60))
sm_sess = sagemaker.Session(boto_session=boto_sess)

# ---- Upload training data folder to S3 (skip if already uploaded) ----
def upload_folder_to_s3(local_folder: str, bucket: str, prefix: str) -> str:
    local = Path(local_folder)
    if not local.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {local_folder}")

    for p in local.rglob("*.csv"):
        rel = p.relative_to(local)
        key = f"{prefix.rstrip('/')}/{rel.as_posix()}"
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            print(f"[SKIP] s3://{bucket}/{key}")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "404":
                raise
            print(f"[UPLOAD] {p} -> s3://{bucket}/{key}")
            s3_client.upload_file(str(p), bucket, key)
    return f"s3://{bucket}/{prefix.rstrip('/')}/"

s3_data_path = upload_folder_to_s3(DATA_DIR, BUCKET, S3_PREFIX)
print(f"Data uploaded to: {s3_data_path}")

metric_defs = [{"Name": "val_loss", "Regex": r"val_loss=([0-9.+-eE]+)"}]

estimator = PyTorch(
    entry_point="aws_train_model.py",
    source_dir="src",
    role=ROLE_ARN,
    framework_version="2.0.0",
    py_version="py310",
    instance_type=INSTANCE_TYPE,            # can be small now (e.g., ml.m5.xlarge or g4dn.xlarge)
    instance_count=1,
    metric_definitions=metric_defs,
    enable_sagemaker_metrics=True,
    environment={"PYTHONUNBUFFERED": "1"},
    sagemaker_session=sm_sess,
    hyperparameters={
        "epochs": 10,
        "window_size": 150,
        "val_months": 1,
        "batch_size": 512,
        "accumulate": 2,                   # effective batch ~512
    },
)

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="val_loss",
    metric_definitions=metric_defs,
    hyperparameter_ranges={
        "hidden_size": IntegerParameter(32, 256),
        "dropout": ContinuousParameter(0.0, 0.6),
        # "num_layers": IntegerParameter(1, 4),
        # "bidirectional": CategoricalParameter([True, False]),
        # "lr": ContinuousParameter(1e-4, 3e-3),
    },
    max_jobs=TUNER_MAX_JOBS,
    max_parallel_jobs=TUNER_MAX_PARALLEL,
    objective_type="Minimize",
)

print("Starting SageMaker Hyperparameter Tuning job...")
tuner.fit({"train": s3_data_path})
tuner.wait()
print("✅ Hyperparameter Tuning job finished.")

# (Optional) Deploy the best model
predictor = tuner.deploy(
    initial_instance_count=1,
    instance_type=ENDPOINT_INSTANCE_TYPE,
    entry_point="inference.py",
)
print(f"✅ Best model deployed. Endpoint: {predictor.endpoint_name}")
