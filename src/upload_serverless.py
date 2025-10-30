#!/usr/bin/env python3
"""
Deploy a trained PyTorch model (inference.py) to a SageMaker Serverless endpoint.

Reads configuration from .env:
  - MODEL_S3_PATH: s3://.../model.tar.gz
  - SAGEMAKER_ROLE_ARN, AWS_REGION
  - ENDPOINT_NAME (optional)
  - FRAMEWORK_VERSION, PY_VERSION (optional)
  - SERVERLESS_MEMORY_MB, SERVERLESS_MAX_CONCURRENCY (optional)
"""

import os
import boto3
import sagemaker
from pathlib import Path
from dotenv import load_dotenv
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def main():
    load_dotenv()

    region = os.getenv("AWS_REGION", "us-east-1")
    role = os.getenv("SAGEMAKER_ROLE_ARN")
    model_s3 = os.getenv("MODEL_S3_PATH")
    endpoint_name = os.getenv("ENDPOINT_NAME", "pytorch-serverless-endpoint")

    if not role:
        raise RuntimeError("SAGEMAKER_ROLE_ARN not set in .env")
    if not model_s3:
        raise RuntimeError("MODEL_S3_PATH not set in .env (s3://.../model.tar.gz)")

    fw = os.getenv("FRAMEWORK_VERSION", "2.3")
    py = os.getenv("PY_VERSION", "py311")

    mem_mb = env_int("SERVERLESS_MEMORY_MB", 2048)
    max_conc = env_int("SERVERLESS_MAX_CONCURRENCY", 10)

    boto_sess = boto3.Session(region_name=region)
    sm_sess = sagemaker.Session(boto_session=boto_sess)

    script_dir = Path(__file__).resolve().parent

    print("\n=== Serverless Deployment Config ===")
    print(f"Region:            {region}")
    print(f"Role:              {role}")
    print(f"Model S3:          {model_s3}")
    print(f"Endpoint Name:     {endpoint_name}")
    print(f"Framework/Python:  {fw} / {py}")
    print(f"Serverless:        {mem_mb} MB, max_concurrency={max_conc}")

    model = PyTorchModel(
        model_data=model_s3,
        role=role,
        entry_point="inference.py",
        source_dir=str(script_dir),
        framework_version=fw,
        py_version=py,
        sagemaker_session=sm_sess,
    )

    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=mem_mb,
        max_concurrency=max_conc,
    )

    print("\nDeploying serverless endpoint...")
    predictor = model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=endpoint_name,
    )
    print(f"Endpoint deployed: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()

