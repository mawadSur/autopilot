import os
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter

def upload_to_s3(local_folder, s3_bucket, s3_prefix):
    """Uploads a directory to an S3 bucket."""
    s3 = boto3.client('s3')
    for dirpath, _, filenames in os.walk(local_folder):
        for f in filenames:
            if f.endswith('.csv'):
                local_path = os.path.join(dirpath, f)
                s3_key = os.path.join(s3_prefix, f)
                print(f"Uploading {local_path} to s3://{s3_bucket}/{s3_key}")
                s3.upload_file(local_path, s3_bucket, s3_key)
    return f"s3://{s3_bucket}/{s3_prefix}"

# ==============================================================================
# 1. --- CONFIGURATION ---
# ==============================================================================
# Replace with your S3 bucket and IAM role name
# sagemaker_session = sagemaker.Session()
S3_BUCKET_NAME = "sagemaker-us-east-1-469090608362"
IAM_ROLE_NAME = "SageMakerExecutionRole"
# ==============================================================================

# Get SageMaker execution role ARN
sagemaker_session = sagemaker.Session()
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName=IAM_ROLE_NAME)['Role']['Arn']

# Define S3 paths
s3_prefix = "eth-price-prediction"
data_folder = "eth_1m_data"
s3_data_path = upload_to_s3(data_folder, S3_BUCKET_NAME, s3_prefix)
print(f"Data uploaded to: {s3_data_path}")

# 2. --- DEFINE THE BASE ESTIMATOR ---
# This estimator defines the core training job configuration.
pytorch_estimator = PyTorch(
    entry_point='aws_train_model.py',
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge', # GPU instance for faster training
    framework_version='1.13',      # Use a specific PyTorch version
    py_version='py39',
    hyperparameters={
        'epochs': 50, # Can be lower for tuning jobs to save time
        'batch-size': 1024,
        'window-size': 150,
        'lookahead-period': 10
    }
)

# 3. --- CONFIGURE THE HYPERPARAMETER TUNER ---
# Define the range of hyperparameters you want to test.
hyperparameter_ranges = {
    'learning-rate': ContinuousParameter(0.0001, 0.01),
    'dropout_rate': ContinuousParameter(0.2, 0.6) # Passed to your model via args
}

# Define the objective metric to optimize.
# This must match a metric printed during training (e.g., "Val Loss: 0.1234")
objective_metric_name = 'validation:loss'
metric_definitions = [{'Name': objective_metric_name, 'Regex': 'Val Loss: ([0-9\\.]+)'}]

# Create the HyperparameterTuner object.
tuner = HyperparameterTuner(
    estimator=pytorch_estimator,
    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=metric_definitions,
    max_jobs=10,  # Total number of training jobs to run
    max_parallel_jobs=1,  # Number of jobs to run in parallel
    objective_type='Minimize'
)

# 4. --- LAUNCH THE TUNING JOB ---
print("\nStarting SageMaker Hyperparameter Tuning job...")
tuner.fit({'train': s3_data_path})
tuner.wait()
print("✅ Hyperparameter Tuning job finished.")

# 5. --- DEPLOY THE BEST MODEL ---
print("\nDeploying the best model found by the tuning job...")
# The tuner object knows which training job was best and deploys that model.
predictor = tuner.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    entry_point='inference.py' # Use our dedicated inference script
)

print(f"✅ Best model deployed successfully. Endpoint name is: {predictor.endpoint_name}")