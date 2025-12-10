#!/usr/bin/env python3
"""
Upload XFUND dataset to Cloudflare R2 storage.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv
from tqdm import tqdm


class CloudflareR2Uploader:
    """Uploads files to Cloudflare R2 using S3-compatible API."""

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        endpoint_url: str,
        bucket_name: str,
        _token: str | None = None,
    ):
        """Initialize R2 client.

        Args:
            access_key_id: R2 Access Key ID
            secret_access_key: R2 Secret Access Key
            endpoint_url: Jurisdiction-specific endpoint (e.g., https://<account_id>.r2.cloudflarestorage.com)
            bucket_name: R2 bucket name
            _token: Optional R2 token (unused)
        """
        self.bucket_name = bucket_name

        # Configure boto3 client for R2
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

        self.s3_client = session.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="auto",  # R2 uses 'auto' region
        )

    def upload_file(self, local_path: Path, r2_key: str, extra_args: dict | None = None) -> bool:
        """Upload a single file to R2.

        Args:
            local_path: Local file path
            r2_key: R2 object key (path in bucket)
            extra_args: Additional S3 upload arguments

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.upload_file(
                str(local_path), self.bucket_name, r2_key, ExtraArgs=extra_args or {}
            )
        except Exception as e:
            print(f"Failed to upload {local_path}: {e}")
            return False
        else:
            return True

    def upload_directory(
        self,
        local_dir: Path,
        r2_prefix: str = "",
        file_extensions: list | None = None,
    ) -> tuple[int, int]:
        """Upload entire directory to R2.

        Args:
            local_dir: Local directory to upload
            r2_prefix: Prefix for R2 keys
            file_extensions: List of file extensions to include (e.g., ['.png', '.json'])

        Returns:
            Tuple of (successful_uploads, failed_uploads)
        """
        if not local_dir.exists():
            print(f"Directory {local_dir} does not exist")
            return 0, 0

        # Find all files to upload
        files_to_upload = []
        for file_path in local_dir.rglob("*"):
            if file_path.is_file() and (
                file_extensions is None or file_path.suffix.lower() in file_extensions
            ):
                # Create R2 key preserving directory structure
                relative_path = file_path.relative_to(local_dir)
                r2_key = f"{r2_prefix}/{relative_path}".strip("/")
                files_to_upload.append((file_path, r2_key))

        print(f"Found {len(files_to_upload)} files to upload")

        successful = 0
        failed = 0

        # Upload with progress bar
        for local_path, r2_key in tqdm(files_to_upload, desc="Uploading"):
            if self.upload_file(local_path, r2_key):
                successful += 1
            else:
                failed += 1

        return successful, failed

    def list_bucket_contents(self, prefix: str = "") -> list:
        """List contents of R2 bucket with optional prefix."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception as e:
            print(f"Failed to list bucket contents: {e}")
            return []


def main():
    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(description="Upload XFUND data to Cloudflare R2")
    parser.add_argument(
        "--access-key-id",
        default=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        help="R2 Access Key ID (or set CLOUDFLARE_R2_ACCESS_KEY_ID env var)",
    )
    parser.add_argument(
        "--secret-access-key",
        default=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
        help="R2 Secret Access Key (or set CLOUDFLARE_R2_SECRET_ACCESS_KEY env var)",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.getenv("CLOUDFLARE_R2_ENDPOINT_URL"),
        help="R2 endpoint URL (or set CLOUDFLARE_R2_ENDPOINT_URL env var)",
    )
    parser.add_argument(
        "--bucket-name",
        default=os.getenv("CLOUDFLARE_R2_BUCKET_NAME"),
        help="R2 bucket name (or set CLOUDFLARE_R2_BUCKET_NAME env var)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("CLOUDFLARE_R2_TOKEN"),
        help="Optional R2 token (or set CLOUDFLARE_R2_TOKEN env var)",
    )
    parser.add_argument("--data-dir", default="data/processed/xfund", help="Local data directory")
    parser.add_argument("--r2-prefix", default="xfund-dataset", help="R2 prefix for uploaded files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--auto-create-bucket",
        action="store_true",
        default=True,
        help="Automatically create bucket if it does not exist (default: True)",
    )

    args = parser.parse_args()

    # Check required arguments
    if not args.access_key_id:
        print(
            "❌ R2 Access Key ID is required (--access-key-id or CLOUDFLARE_R2_ACCESS_KEY_ID env var)"
        )
        return 1
    if not args.secret_access_key:
        print(
            "❌ R2 Secret Access Key is required (--secret-access-key or CLOUDFLARE_R2_SECRET_ACCESS_KEY env var)"
        )
        return 1
    if not args.endpoint_url:
        print(
            "❌ R2 Endpoint URL is required (--endpoint-url or CLOUDFLARE_R2_ENDPOINT_URL env var)"
        )
        return 1
    if not args.bucket_name:
        print("❌ R2 Bucket Name is required (--bucket-name or CLOUDFLARE_R2_BUCKET_NAME env var)")
        return 1

    # Initialize uploader
    uploader = CloudflareR2Uploader(
        access_key_id=args.access_key_id,
        secret_access_key=args.secret_access_key,
        endpoint_url=args.endpoint_url,
        bucket_name=args.bucket_name,
        token=args.token,
    )

    # Auto-create bucket if requested and required environment variables are available
    if args.auto_create_bucket and not args.dry_run:
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        api_token = os.getenv("CLOUDFLARE_ACCOUNT_API_TOKEN")

        if account_id and api_token:
            print(f"🔍 Ensuring bucket '{args.bucket_name}' exists...")
            # subprocess and sys imported at module level

            # Call the manage script to ensure bucket exists
            result = subprocess.run(  # noqa: S603
                [sys.executable, "scripts/manage_r2_bucket.py", "auto-ensure"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                print(f"⚠️  Could not auto-create bucket: {result.stderr.strip()}")
                print("Continuing with upload attempt...")
        else:
            print(
                "⚠️  Auto-create bucket requested but CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_ACCOUNT_API_TOKEN not available"
            )

    data_dir = Path(args.data_dir)

    if args.dry_run:
        print("DRY RUN - Files that would be uploaded:")
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(data_dir)
                r2_key = f"{args.r2_prefix}/{relative_path}"
                print(f"  {file_path} -> {r2_key}")
        return None

    print(f"Uploading {data_dir} to R2 bucket '{args.bucket_name}'...")

    # Upload VAE data
    vae_dir = data_dir / "vae"
    if vae_dir.exists():
        print("Uploading VAE training data...")
        successful, failed = uploader.upload_directory(vae_dir, f"{args.r2_prefix}/vae")
        print(f"VAE upload: {successful} successful, {failed} failed")

    # Upload UNET data
    unet_dir = data_dir / "unet"
    if unet_dir.exists():
        print("Uploading UNET training data...")
        successful, failed = uploader.upload_directory(unet_dir, f"{args.r2_prefix}/unet")
        print(f"UNET upload: {successful} successful, {failed} failed")

    # Upload raw data if exists
    raw_dir = Path("data/raw/xfund")
    if raw_dir.exists():
        print("Uploading raw XFUND data...")
        successful, failed = uploader.upload_directory(
            raw_dir,
            f"{args.r2_prefix}/raw",
            file_extensions=[".json", ".zip"],  # Only upload JSON and ZIP files
        )
        print(f"Raw data upload: {successful} successful, {failed} failed")

    print("Upload completed!")
    return None


if __name__ == "__main__":
    main()
