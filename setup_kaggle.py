#!/usr/bin/env python3
"""
Kaggle API Setup Script

This script helps set up the Kaggle API credentials for downloading ImageNet-1k dataset.
It provides instructions and can optionally validate the setup.

Usage:
    python setup_kaggle.py --validate
    python setup_kaggle.py --setup-credentials /path/to/kaggle.json
"""

import os
import sys
import json
import argparse
from pathlib import Path


def print_setup_instructions():
    """Print detailed setup instructions for Kaggle API."""
    print("=" * 60)
    print("KAGGLE API SETUP INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. Get your Kaggle API credentials:")
    print("   - Go to https://www.kaggle.com/")
    print("   - Sign in to your account")
    print("   - Click on your profile picture → Account")
    print("   - Scroll down to 'API' section")
    print("   - Click 'Create New API Token'")
    print("   - This downloads 'kaggle.json' file")
    print()
    print("2. Set up credentials on your EC2 instance:")
    print("   mkdir -p ~/.kaggle")
    print("   mv kaggle.json ~/.kaggle/")
    print("   chmod 600 ~/.kaggle/kaggle.json")
    print()
    print("3. Verify setup:")
    print("   python setup_kaggle.py --validate")
    print()
    print("4. Download ImageNet dataset:")
    print("   python data_transformer.py --output-dir /path/to/imagenet")
    print()


def validate_kaggle_setup():
    """Validate Kaggle API setup."""
    print("Validating Kaggle API setup...")
    
    # Check if kaggle.json exists
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("❌ kaggle.json not found in ~/.kaggle/")
        print("Please follow the setup instructions above.")
        return False
    
    # Check file permissions
    if oct(kaggle_json.stat().st_mode)[-3:] != "600":
        print("⚠️  kaggle.json permissions should be 600")
        print("Run: chmod 600 ~/.kaggle/kaggle.json")
    
    # Validate JSON content
    try:
        with open(kaggle_json, 'r') as f:
            creds = json.load(f)
        
        required_keys = ['username', 'key']
        if not all(key in creds for key in required_keys):
            print("❌ kaggle.json missing required keys (username, key)")
            return False
        
        if not creds['username'] or not creds['key']:
            print("❌ kaggle.json has empty username or key")
            return False
        
        print("✅ kaggle.json found and valid")
        
    except json.JSONDecodeError:
        print("❌ kaggle.json is not valid JSON")
        return False
    except Exception as e:
        print(f"❌ Error reading kaggle.json: {e}")
        return False
    
    # Test Kaggle API
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        print("✅ Kaggle API authentication successful")
        
        # Test API call
        competitions = api.competitions_list()
        print(f"✅ API working - found {len(competitions)} competitions")
        
        return True
        
    except ImportError:
        print("❌ Kaggle API not installed")
        print("Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Kaggle API authentication failed: {e}")
        return False


def setup_credentials(credentials_path: str):
    """Set up Kaggle credentials from provided file."""
    credentials_file = Path(credentials_path)
    
    if not credentials_file.exists():
        print(f"❌ Credentials file not found: {credentials_path}")
        return False
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Copy credentials file
    kaggle_json = kaggle_dir / "kaggle.json"
    try:
        import shutil
        shutil.copy2(credentials_file, kaggle_json)
        os.chmod(kaggle_json, 0o600)
        print(f"✅ Credentials copied to {kaggle_json}")
        return True
    except Exception as e:
        print(f"❌ Failed to copy credentials: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup and validate Kaggle API for ImageNet download"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate Kaggle API setup"
    )
    parser.add_argument(
        "--setup-credentials",
        type=str,
        help="Path to kaggle.json file to set up"
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Show setup instructions"
    )
    
    args = parser.parse_args()
    
    if args.instructions or (not args.validate and not args.setup_credentials):
        print_setup_instructions()
        return
    
    if args.setup_credentials:
        success = setup_credentials(args.setup_credentials)
        if success:
            print("Now run: python setup_kaggle.py --validate")
        sys.exit(0 if success else 1)
    
    if args.validate:
        success = validate_kaggle_setup()
        if success:
            print("\n🎉 Kaggle API setup complete!")
            print("You can now run the ImageNet downloader:")
            print("python data_transformer.py --output-dir /path/to/imagenet")
        else:
            print("\n❌ Kaggle API setup incomplete")
            print("Run: python setup_kaggle.py --instructions")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


