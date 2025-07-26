import os
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_to_s3(local_folder, s3_bucket, s3_prefix):
    """
    Uploads a directory to an S3 bucket, skipping files that already exist.
    """
    s3 = boto3.client('s3')
    if not os.path.isdir(local_folder):
        logger.error(f"🚫 Error: Local directory '{local_folder}' not found. Halting.")
        exit()
    local_files = [f for f in os.listdir(local_folder) if f.endswith('.csv')]
    if not local_files:
        logger.error(f"🚫 Error: No CSV files found in '{local_folder}'. Halting.")
        exit()
    logger.info(f"Found {len(local_files)} local CSV files for potential upload.")
    for filename in local_files:
        local_path = os.path.join(local_folder, filename)
        s3_key = os.path.join(s3_prefix, filename)
        try:
            s3.head_object(Bucket=s3_bucket, Key=s3_key)
            logger.info(f"✅ File already exists on S3, skipping: s3://{s3_bucket}/{s3_key}")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"Uploading {local_path} to s3://{s3_bucket}/{s3_key}")
                s3.upload_file(local_path, s3_bucket, s3_key)
            else:
                raise
    return f"s3://{s3_bucket}/{s3_prefix}"

def cleanup_sagemaker_jobs(tuner, sagemaker_session):
    """
    Stops the parent tuning job, which in turn stops all its child training jobs.
    """
    try:
        tuner_job_name = tuner.latest_tuning_job.name
        logger.warning(f"🛑 Stopping parent tuning job: {tuner_job_name}")
        
        # --- KEY FIX: Stop the parent tuning job ---
        # This will automatically stop all running child jobs.
        sagemaker_session.sagemaker_client.stop_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuner_job_name
        )
        logger.info("   - Stop command sent to parent tuning job.")
        
    except Exception as e:
        logger.error(f"An error occurred during cleanup: {e}")

# ==============================================================================
# 1. --- CONFIGURATION ---
# Use environment variables for flexibility instead of hardcoding.
# ==============================================================================
S3_BUCKET_NAME = "sagemaker-us-east-1-469090608362" #
IAM_ROLE_NAME = "SageMakerExecutionRole" #
# ==============================================================================

sagemaker_session = sagemaker.Session()
# --- OPTIMIZED: Simplified and more robust role retrieval ---
try:
    role = sagemaker.get_execution_role()
except ValueError:
    logger.info(f"Could not get execution role. Trying to construct from role name: {IAM_ROLE_NAME}")
    iam_client = boto3.client('iam')
    role_arn = iam_client.get_role(RoleName=IAM_ROLE_NAME)['Role']['Arn']
    role = role_arn

# --- Data Upload ---
s3_prefix = "eth-price-prediction"
data_folder = "eth_1m_data"
s3_data_path = upload_to_s3(data_folder, S3_BUCKET_NAME, s3_prefix)
logger.info(f"✅ Data is ready at: {s3_data_path}")

# 2. --- DEFINE THE BASE ESTIMATOR ---
pytorch_estimator = PyTorch(
    entry_point='aws_train_model.py',
    source_dir='./', # Ensure inference.py is also packaged
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.12xlarge',
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'epochs': 50, 'patience': 5, 'batch-size': 512, 'window-size': 150,
        'lookahead-period': 10, 'risk-reward-ratio': 2.0, 'profit-threshold-pct': 0.5
    }
)

# 3. --- CONFIGURE THE HYPERPARAMETER TUNER ---
hyperparameter_ranges = {
    'learning-rate': ContinuousParameter(0.0001, 0.01, scaling_type='Logarithmic'),
    'dropout-rate': ContinuousParameter(0.2, 0.5)
}

objective_metric_name = 'validation:accuracy'
metric_definitions = [
    {'Name': 'validation:loss', 'Regex': 'Val Loss: ([0-9\\.]+)'},
    {'Name': 'validation:accuracy', 'Regex': 'Val Acc: ([0-9\\.]+)'}
]

tuner = HyperparameterTuner(
    estimator=pytorch_estimator,
    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=metric_definitions,
    max_jobs=10,
    max_parallel_jobs=2,
    objective_type='Maximize',
    # --- OPTIMIZATION: Enable automatic early stopping ---
    # This will stop individual training jobs that are not showing improvement, saving time and money.
    early_stopping_type='Auto'
)

# 4. --- LAUNCH AND MANAGE THE TUNING JOB ---
try:
    logger.info("\n🚀 Starting SageMaker Hyperparameter Tuning job...")
    tuner.fit({'train': s3_data_path}, wait=True, logs="All")
    logger.info("✅ Tuning job finished successfully.")

    # 5. --- DEPLOY THE BEST MODEL ---
    logger.info("Deploying the best model from the tuning job...")
    # Make sure to pass the inference script to the deploy method
    predictor = tuner.deploy(
        initial_instance_count=1,
        instance_type='ml.g4dn.12xlarge',
        entry_point='inference.py' 
    )
    logger.info(f"✅ Best model deployed. Endpoint name: {predictor.endpoint_name}")

except KeyboardInterrupt:
    logger.warning("\n🛑 User interrupted the process. Initiating cleanup...")
    cleanup_sagemaker_jobs(tuner, sagemaker_session)
    logger.info("Cleanup complete. Exiting.")

except Exception as e:
    logger.error(f"An unexpected error occurred: {e}")
    logger.warning("Attempting to clean up any running jobs...")
    cleanup_sagemaker_jobs(tuner, sagemaker_session)