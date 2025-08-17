import os
from dotenv import load_dotenv

load_dotenv()

MODEL_S3_PATH = os.getenv('MODEL_S3_PATH')
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME") # No default needed if it's in .env
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
bar = os.getenv('FOO')

print(f"MODEL_S3_PATH: {MODEL_S3_PATH}")
print(f"ENDPOINT_NAME: {ENDPOINT_NAME}")
print(f"ROLE_ARN: {ROLE_ARN}")
print(f"FOO: {bar}")