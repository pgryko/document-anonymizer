#!/usr/bin/env python3
"""
Manage Cloudflare R2 buckets and tokens using the Cloudflare API.
"""

import argparse
import os
import sys

import requests
from dotenv import load_dotenv

# Constants
REQUEST_TIMEOUT = 30  # seconds
BUCKET_EXISTS_ERROR_CODE = 10014


class UnsupportedHTTPMethodError(ValueError):
    """Raised when an unsupported HTTP method is used."""


class CloudflareR2Manager:
    """Manages R2 buckets and tokens via Cloudflare API."""

    def __init__(self, account_id: str, api_token: str):
        """Initialize with Cloudflare credentials."""
        self.account_id = account_id
        self.api_token = api_token
        self.base_url = "https://api.cloudflare.com/client/v4"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def _make_request(self, method: str, endpoint: str, data: dict | None = None) -> dict:
        """Make authenticated request to Cloudflare API."""
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
            elif method.upper() == "POST":
                response = requests.post(
                    url, headers=self.headers, json=data, timeout=REQUEST_TIMEOUT
                )
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
            else:
                raise UnsupportedHTTPMethodError(method)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e.response, "text"):
                print(f"Response: {e.response.text}")
            sys.exit(1)

    def list_buckets(self) -> list[dict]:
        """List all R2 buckets in the account."""
        response = self._make_request("GET", f"/accounts/{self.account_id}/r2/buckets")

        if response.get("success"):
            return response.get("result", [])
        print(f"Failed to list buckets: {response.get('errors', [])}")
        return []

    def create_bucket(self, bucket_name: str, location_hint: str = "WNAM") -> dict:
        """Create a new R2 bucket.

        Args:
            bucket_name: Name of the bucket to create
            location_hint: Location hint (e.g., WNAM, ENAM, APAC, etc.)

        Returns:
            Bucket creation response
        """
        data = {"name": bucket_name, "locationHint": location_hint}

        response = self._make_request("POST", f"/accounts/{self.account_id}/r2/buckets", data)

        if response.get("success"):
            print(f"✅ Bucket '{bucket_name}' created successfully")
            return response.get("result", {})
        errors = response.get("errors", [])
        for error in errors:
            if error.get("code") == BUCKET_EXISTS_ERROR_CODE:  # Bucket already exists
                print(f"⚠️  Bucket '{bucket_name}' already exists")
                return {"name": bucket_name, "already_exists": True}
            print(f"❌ Failed to create bucket: {error}")
        return {}

    def delete_bucket(self, bucket_name: str) -> bool:
        """Delete an R2 bucket."""
        response = self._make_request(
            "DELETE", f"/accounts/{self.account_id}/r2/buckets/{bucket_name}"
        )

        if response.get("success"):
            print(f"✅ Bucket '{bucket_name}' deleted successfully")
            return True
        print(f"❌ Failed to delete bucket: {response.get('errors', [])}")
        return False

    def list_tokens(self) -> list[dict]:
        """List all R2 tokens in the account."""
        response = self._make_request("GET", f"/accounts/{self.account_id}/r2/credentials")

        if response.get("success"):
            return response.get("result", [])
        print(f"Failed to list tokens: {response.get('errors', [])}")
        return []

    def create_token(
        self,
        token_name: str,
        permissions: list[str] | None = None,
        bucket_names: list[str] | None = None,
        ttl_days: int | None = None,
    ) -> dict:
        """Create a new R2 token.

        Args:
            token_name: Name for the token
            permissions: List of permissions ('admin', 'read', 'edit')
            bucket_names: List of bucket names to scope to (None for all buckets)
            ttl_days: Token TTL in days (None for no expiration)

        Returns:
            Token creation response
        """
        if permissions is None:
            permissions = ["admin"]

        data = {
            "name": token_name,
            "maxDurationSeconds": ttl_days * 86400 if ttl_days else None,
            "permissions": {
                "admin": "admin" in permissions,
                "read": "read" in permissions,
                "edit": "edit" in permissions,
            },
        }

        # Add bucket scope if specified
        if bucket_names:
            data["buckets"] = bucket_names

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        response = self._make_request("POST", f"/accounts/{self.account_id}/r2/credentials", data)

        if response.get("success"):
            result = response.get("result", {})
            print(f"✅ Token '{token_name}' created successfully")
            print(f"   Access Key ID: {result.get('accessKeyId')}")
            print(f"   Secret Access Key: {result.get('secretAccessKey')}")
            print("   ⚠️  Save these credentials securely - they won't be shown again!")
            return result
        print(f"❌ Failed to create token: {response.get('errors', [])}")
        return {}

    def delete_token(self, access_key_id: str) -> bool:
        """Delete an R2 token by access key ID."""
        response = self._make_request(
            "DELETE", f"/accounts/{self.account_id}/r2/credentials/{access_key_id}"
        )

        if response.get("success"):
            print(f"✅ Token '{access_key_id}' deleted successfully")
            return True
        print(f"❌ Failed to delete token: {response.get('errors', [])}")
        return False

    def ensure_bucket_exists(self, bucket_name: str, location_hint: str = "WNAM") -> bool:
        """Ensure a bucket exists, creating it if necessary.

        Args:
            bucket_name: Name of the bucket to ensure exists
            location_hint: Location hint for bucket creation

        Returns:
            True if bucket exists or was created, False on error
        """
        # Check if bucket already exists
        buckets = self.list_buckets()
        existing_bucket = next((b for b in buckets if b.get("name") == bucket_name), None)

        if existing_bucket:
            print(f"✅ Bucket '{bucket_name}' already exists")
            return True

        # Create the bucket
        result = self.create_bucket(bucket_name, location_hint)
        return bool(result)


def main():
    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(description="Manage Cloudflare R2 buckets and tokens")
    parser.add_argument(
        "--account-id",
        default=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        help="Cloudflare Account ID",
    )
    parser.add_argument(
        "--api-token",
        default=os.getenv("CLOUDFLARE_ACCOUNT_API_TOKEN"),
        help="Cloudflare API Token",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Bucket commands
    bucket_parser = subparsers.add_parser("bucket", help="Bucket management")
    bucket_subparsers = bucket_parser.add_subparsers(dest="bucket_action")

    bucket_subparsers.add_parser("list", help="List all buckets")

    create_bucket_parser = bucket_subparsers.add_parser("create", help="Create a bucket")
    create_bucket_parser.add_argument("name", help="Bucket name")
    create_bucket_parser.add_argument(
        "--location", default="WNAM", help="Location hint (WNAM, ENAM, APAC, etc.)"
    )

    delete_bucket_parser = bucket_subparsers.add_parser("delete", help="Delete a bucket")
    delete_bucket_parser.add_argument("name", help="Bucket name")

    # Token commands
    token_parser = subparsers.add_parser("token", help="Token management")
    token_subparsers = token_parser.add_subparsers(dest="token_action")

    token_subparsers.add_parser("list", help="List all tokens")

    create_token_parser = token_subparsers.add_parser("create", help="Create a token")
    create_token_parser.add_argument("name", help="Token name")
    create_token_parser.add_argument(
        "--permissions",
        nargs="+",
        choices=["admin", "read", "edit"],
        default=["admin"],
        help="Token permissions",
    )
    create_token_parser.add_argument("--buckets", nargs="+", help="Scope to specific buckets")
    create_token_parser.add_argument("--ttl-days", type=int, help="Token TTL in days")

    delete_token_parser = token_subparsers.add_parser("delete", help="Delete a token")
    delete_token_parser.add_argument("access_key_id", help="Access Key ID of token to delete")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Quick setup for document-anonymizer")
    setup_parser.add_argument(
        "--bucket-name",
        default="document-anonymizer-data",
        help="Bucket name to create",
    )
    setup_parser.add_argument(
        "--token-name", default="document-anonymizer-token", help="Token name to create"
    )

    # Auto-ensure command
    auto_ensure_parser = subparsers.add_parser(
        "auto-ensure", help="Auto-create bucket from .env if it does not exist"
    )
    auto_ensure_parser.add_argument(
        "--location",
        default="WNAM",
        help="Location hint for bucket creation (WNAM, ENAM, APAC, etc.)",
    )

    args = parser.parse_args()

    if not args.account_id:
        print("❌ CLOUDFLARE_ACCOUNT_ID is required (set via --account-id or environment variable)")
        sys.exit(1)

    if not args.api_token:
        print(
            "❌ CLOUDFLARE_ACCOUNT_API_TOKEN is required (set via --api-token or environment variable)"
        )
        sys.exit(1)

    manager = CloudflareR2Manager(args.account_id, args.api_token)

    if args.command == "bucket":
        if args.bucket_action == "list":
            buckets = manager.list_buckets()
            if buckets:
                print("📦 R2 Buckets:")
                for bucket in buckets:
                    print(f"   • {bucket.get('name')} (created: {bucket.get('creation_date')})")
            else:
                print("No buckets found")

        elif args.bucket_action == "create":
            manager.create_bucket(args.name, args.location)

        elif args.bucket_action == "delete":
            manager.delete_bucket(args.name)

    elif args.command == "token":
        if args.token_action == "list":
            tokens = manager.list_tokens()
            if tokens:
                print("🔑 R2 Tokens:")
                for token in tokens:
                    print(f"   • {token.get('name')} (ID: {token.get('accessKeyId')})")
            else:
                print("No tokens found")

        elif args.token_action == "create":
            manager.create_token(args.name, args.permissions, args.buckets, args.ttl_days)

        elif args.token_action == "delete":
            manager.delete_token(args.access_key_id)

    elif args.command == "setup":
        print("🚀 Setting up R2 for document-anonymizer...")

        # Create bucket
        bucket_result = manager.create_bucket(args.bucket_name)

        # Create token
        if bucket_result:
            token_result = manager.create_token(
                args.token_name, permissions=["admin"], bucket_names=[args.bucket_name]
            )

            if token_result:
                print("\n📋 Add these to your .env file:")
                print(f"CLOUDFLARE_R2_ACCESS_KEY_ID={token_result.get('accessKeyId')}")
                print(f"CLOUDFLARE_R2_SECRET_ACCESS_KEY={token_result.get('secretAccessKey')}")
                print(f"CLOUDFLARE_R2_BUCKET_NAME={args.bucket_name}")

    elif args.command == "auto-ensure":
        # Get bucket name from environment
        bucket_name = os.getenv("CLOUDFLARE_R2_BUCKET_NAME")

        if not bucket_name:
            print("❌ CLOUDFLARE_R2_BUCKET_NAME environment variable is required for auto-ensure")
            sys.exit(1)

        print(f"🔍 Ensuring bucket '{bucket_name}' exists...")
        success = manager.ensure_bucket_exists(bucket_name, args.location)

        if success:
            print(f"✅ Bucket '{bucket_name}' is ready to use!")
        else:
            print(f"❌ Failed to ensure bucket '{bucket_name}' exists")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
