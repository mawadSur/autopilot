import os
import sagemaker
from sagemaker.pytorch import PyTorchModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---

# 1. PASTE THE S3 PATH TO YOUR MODEL ARTIFACT HERE
model_s3_path = "s3://sagemaker-us-east-1-469090608362/pytorch-training-250726-0525-010-0e6c69e0/output/model.tar.gz"

# 2. CHOOSE A NAME FOR YOUR ENDPOINT
endpoint_name = "eth-price-prediction-endpoint-v1"

# 3. SPECIFY YOUR AWS ACCOUNT ID AND ROLE NAME
#    Replace 'YOUR_AWS_ACCOUNT_ID' with your actual 12-digit account number.
AWS_ACCOUNT_ID = "469090608362" # Example: "123456789012"
IAM_ROLE_NAME = "SageMakerExecutionRole"

# --- KEY FIX: Construct the role ARN directly ---
# This is the robust way to specify the role when running from a local machine.
role = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{IAM_ROLE_NAME}"
logger.info(f"Using IAM Role ARN: {role}")


# --- Create a SageMaker PyTorchModel object ---
# This points to your trained model data and the inference code.
pytorch_model = PyTorchModel(
    model_data=model_s3_path,
    role=role,
    entry_point='inference.py',   # Your inference script
    framework_version='2.0.0',    # Must match the training script
    py_version='py310'            # Must match the training script
)

logger.info("✅ PyTorchModel object created.")

# --- Deploy the model to an endpoint ---
logger.info(f"🚀 Deploying model to endpoint: {endpoint_name}...")

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium', # Use a cost-effective instance type for inference
    endpoint_name=endpoint_name
)

logger.info(f"✅ Deployment successful. Endpoint is live at: {predictor.endpoint_name}")