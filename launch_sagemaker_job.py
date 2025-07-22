import os
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter
from botocore.exceptions import ClientError

def upload_to_s3(local_folder, s3_bucket, s3_prefix):
    """
    Uploads a directory to an S3 bucket, skipping files that already exist.
    """
    s3 = boto3.client('s3')
    
    for dirpath, _, filenames in os.walk(local_folder):
        for f in filenames:
            if f.endswith('.csv'):
                local_path = os.path.join(dirpath, f)
                s3_key = os.path.join(s3_prefix, f)
                
                try:
                    s3.head_object(Bucket=s3_bucket, Key=s3_key)
                    print(f"File already exists, skipping: s3://{s3_bucket}/{s3_key}")
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        print(f"Uploading {local_path} to s3://{s3_bucket}/{s3_key}")
                        s3.upload_file(local_path, s3_bucket, s3_key)
                    else:
                        raise
                        
    return f"s3://{s3_bucket}/{s3_prefix}"

# ==============================================================================
# 1. --- CONFIGURATION ---
# ==============================================================================
S3_BUCKET_NAME = "sagemaker-us-east-1-469090608362" #
IAM_ROLE_NAME = "SageMakerExecutionRole" #
# ==============================================================================

sagemaker_session = sagemaker.Session()
try:
    role = 'arn:aws:iam::469090608362:role/SageMakerExecutionRole' #
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName=IAM_ROLE_NAME)['Role']['Arn']

s3_prefix = "eth-price-prediction" #
data_folder = "eth_1m_data" #
s3_data_path = upload_to_s3(data_folder, S3_BUCKET_NAME, s3_prefix)
print(f"Data uploaded to: {s3_data_path}")

# 2. --- DEFINE THE BASE ESTIMATOR ---
pytorch_estimator = PyTorch(
    entry_point='aws_train_model.py', #
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge', #
    framework_version='1.13', #
    py_version='py39', #
    hyperparameters={
        'epochs': '50',
        'batch-size': '1024',
        'window-size': '150',
        'lookahead-period': '10',
        'risk-reward-ratio': 2.0,
        'profit-threshold-pct': 0.5
    }
)

# 3. --- CONFIGURE THE HYPERPARAMETER TUNER ---
hyperparameter_ranges = {
    'learning-rate': ContinuousParameter(0.0001, 0.01, scaling_type='Logarithmic'),
    'dropout-rate': ContinuousParameter(0.2, 0.6)
}

# ✅ CORRECTED: Changed objective to maximize accuracy
objective_metric_name = 'validation:accuracy'
metric_definitions = [
    {'Name': 'validation:loss', 'Regex': 'Val Loss: ([0-9\\.]+)'},
    {'Name': 'validation:accuracy', 'Regex': 'Val Acc: ([0-9\\.]+)'} # Added regex for accuracy
]

tuner = HyperparameterTuner(
    estimator=pytorch_estimator,
    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=metric_definitions,
    max_jobs=20,
    max_parallel_jobs=2,
    objective_type='Maximize' # Changed to Maximize for accuracy
)

# 4. --- LAUNCH THE TUNING JOB ---
print("\n🚀 Starting SageMaker Hyperparameter Tuning job...")
print(f"   - Max Jobs: {tuner.max_jobs}")
print(f"   - Parallel Jobs: {tuner.max_parallel_jobs}")
print(f"   - Objective: {tuner.objective_type} {tuner.objective_metric_name}") # Updated print statement
print("-" * 40)

tuner.fit({'train': s3_data_path}, wait=True, logs=True)

best_job_name = tuner.best_training_job()
print(f"🏆 Best training job found: {best_job_name}")

best_estimator = PyTorch.attach(best_job_name)

predictor = best_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    entry_point='inference.py'
)

print(f"✅ Best model deployed successfully. Endpoint name is: {predictor.endpoint_name}")