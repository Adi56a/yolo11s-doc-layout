from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import NoCredentialsError
except ImportError:
    boto3 = None
    NoCredentialsError = Exception


class S3StorageHelper:
    """
    A class to handle uploading files to AWS S3, with a robust local directory
    fallback for development and offline testing.
    """
    def __init__(
        self,
        bucket_name: str | None = None,
        region: str | None = None,
        use_local_storage: bool = True,
        local_storage_dir: str | Path = "outputs/local_s3_mock",
    ) -> None:
        self.use_local_storage = use_local_storage
        self.local_storage_dir = Path(local_storage_dir)

        if self.use_local_storage:
            self.local_storage_dir.mkdir(parents=True, exist_ok=True)
            print(f"[STORAGE] Running in local mock mode. Target dir: {self.local_storage_dir.resolve()}")
        else:
            if boto3 is None:
                raise ImportError(
                    "boto3 library is required when USE_LOCAL_STORAGE=False. "
                    "Please install it using 'pip install boto3'."
                )
            
            # Load credentials and configuration from environment variables if not passed
            self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME", "edtech-b2b-media")
            self.region = region or os.getenv("AWS_REGION", "ap-south-1")
            
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            
            # Initialize boto3 S3 client
            self.s3_client = boto3.client(
                "s3",
                region_name=self.region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
            print(f"[STORAGE] Connected to AWS S3. Bucket: {self.bucket_name}, Region: {self.region}")

    def upload_file(self, local_file_path: str | Path, target_key: str) -> str:
        """
        Uploads a file to S3 or copies it to the local mock directory.
        Returns the final URL.
        """
        local_file_path = Path(local_file_path)
        if not local_file_path.exists():
            raise FileNotFoundError(f"Source file not found: {local_file_path}")

        # Normalize key slashes for S3
        target_key = target_key.replace("\\", "/")

        if self.use_local_storage:
            # Simulate S3 structure locally
            destination_path = self.local_storage_dir / target_key
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_file_path, destination_path)
            
            # Return absolute local file URL (clickable in browsers/terminals)
            return f"file:///{destination_path.resolve().as_posix()}"
        else:
            try:
                # Content type detection based on extension
                content_type = "image/png" if target_key.endswith(".png") else "application/pdf"
                
                # Upload to S3
                self.s3_client.upload_file(
                    Filename=str(local_file_path),
                    Bucket=self.bucket_name,
                    Key=target_key,
                    ExtraArgs={"ContentType": content_type}
                )
                
                # Construct standard public S3 URL matching the edtech-b2b-media template
                return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{target_key}"
            except NoCredentialsError:
                raise RuntimeError("AWS Credentials not found or invalid. Please check your .env configuration.")
            except Exception as e:
                raise RuntimeError(f"Failed S3 upload for key '{target_key}': {e}")
