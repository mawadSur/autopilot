import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import os

# 1. --- Session and Role ---
sagemaker_session = sagemaker.Session()
# Replace with your IAM role ARN if needed, otherwise get_execution_role() works in SageMaker environments
role = 'arn:aws:iam::469090608362:role/SageMakerExecutionRole'
bucket = sagemaker_session.default_bucket()

# 2. --- Conditionally Upload Local Data ---
s3_prefix = 'data/eth-1m-pytorch'
s3_data_path = f's3://{bucket}/{s3_prefix}'
upload_data = True

# Get the absolute path to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Join it with the relative data folder name to create a robust, absolute path
local_data_path = os.path.join(script_dir, 'eth_1m_data')

try:
    # Count local CSV files
    local_files = [f for f in os.listdir(local_data_path) if f.endswith('.csv')]
    local_count = len(local_files)

    # Count files on S3
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    s3_count = response.get('KeyCount', 0)

    print(f"Found {local_count} local files and {s3_count} files on S3.")

    if local_count > 0 and local_count == s3_count:
        print("✅ File counts match. Skipping S3 upload.")
        upload_data = False
    elif local_count == 0:
        print(f"🚫 Error: Local directory '{local_data_path}' is empty. Halting.")
        exit()

except FileNotFoundError:
    print(f"🚫 Error: Local directory '{local_data_path}' not found. Halting.")
    exit()

if upload_data:
    print(f"Uploading data from '{local_data_path}' to S3...")
    s3_data_path = sagemaker_session.upload_data(
        path=local_data_path,
        bucket=bucket,
        key_prefix=s3_prefix
    )
    print(f"✅ Data uploaded to: {s3_data_path}")


# 3. --- Define the SageMaker PyTorch Estimator for GPU Training ---
pytorch_estimator = PyTorch(
    entry_point='aws_train_model.py',
    source_dir='.',  # This will package the entire directory
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    framework_version='2.0.0',
    py_version='py310',
    sagemaker_session=sagemaker_session,
    hyperparameters={
        'epochs': 100,
        'patience': 10,
        'batch-size': 64,
        'learning-rate': 0.001,
        'window-size': 150,
        'lookahead-period': 10
    }
)

# 4. --- Start the Training Job ---
pytorch_estimator.fit(
    {'training': s3_data_path},
    wait=True,
    logs="All"
)

print("✅ Training job finished.")

# 5. --- Deploy the Model to an Endpoint ---
print("\nTraining job finished. Deploying model to a real-time endpoint...")

predictor = pytorch_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    entry_point='inference.py'  # <-- ADD THIS LINE
)

print(f"✅ Model deployed successfully. Endpoint name is: {predictor.endpoint_name}")