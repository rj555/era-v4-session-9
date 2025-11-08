#!/usr/bin/env python3
"""
Example usage of ImageNet-1k dataset downloader

This script demonstrates how to use the ImageNet downloader programmatically
and provides examples for common use cases.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_transformer import ImageNetDownloader


def example_basic_download():
    """Example: Basic dataset download."""
    print("Example 1: Basic Dataset Download")
    print("=" * 40)
    
    # Set output directory
    output_dir = "/Users/maverick/Work/DataSets/imagenet_example"
    
    # Create downloader instance
    downloader = ImageNetDownloader(output_dir=output_dir)
    
    # Run full pipeline
    success = downloader.run_full_pipeline(resume=True, cleanup=True)
    
    if success:
        print("✅ Dataset download completed successfully!")
    else:
        print("❌ Dataset download failed!")
    
    return success


def example_resume_download():
    """Example: Resume interrupted download."""
    print("\nExample 2: Resume Interrupted Download")
    print("=" * 40)
    
    output_dir = "/Users/maverick/Work/DataSets/imagenet_example"
    downloader = ImageNetDownloader(output_dir=output_dir)
    
    # Load existing progress
    progress = downloader._load_progress()
    print(f"Found {len(downloader.downloaded_files)} previously downloaded files")
    
    # Resume download
    success = downloader.download_dataset(resume=True)
    
    if success:
        print("✅ Download resumed successfully!")
    else:
        print("❌ Resume failed!")
    
    return success


def example_validate_only():
    """Example: Validate existing dataset."""
    print("\nExample 3: Validate Existing Dataset")
    print("=" * 40)
    
    output_dir = "/Users/maverick/Work/DataSets/imagenet_example"
    downloader = ImageNetDownloader(output_dir=output_dir)
    
    # Validate dataset
    is_valid = downloader.validate_dataset()
    
    if is_valid:
        print("✅ Dataset validation passed!")
    else:
        print("❌ Dataset validation failed!")
    
    return is_valid


def example_custom_credentials():
    """Example: Using custom Kaggle credentials."""
    print("\nExample 4: Custom Kaggle Credentials")
    print("=" * 40)
    
    # Path to custom kaggle.json
    kaggle_credentials = os.path.expanduser("~/.kaggle/kaggle.json")
    output_dir = "/Users/maverick/Work/DataSets/imagenet_example"
    
    # Create downloader with custom credentials
    downloader = ImageNetDownloader(
        output_dir=output_dir,
        kaggle_credentials_path=kaggle_credentials
    )
    
    # Test authentication
    try:
        downloader._setup_kaggle_api()
        print("✅ Kaggle API authentication successful!")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False


def example_step_by_step():
    """Example: Step-by-step processing."""
    print("\nExample 5: Step-by-Step Processing")
    print("=" * 40)
    
    output_dir = "/Users/maverick/Work/DataSets/imagenet_example"
    downloader = ImageNetDownloader(output_dir=output_dir)
    
    steps = [
        ("Download dataset", downloader.download_dataset),
        ("Extract dataset", downloader.extract_dataset),
        ("Organize dataset", downloader.organize_dataset),
        ("Validate dataset", downloader.validate_dataset),
    ]
    
    for step_name, step_func in steps:
        print(f"\nStep: {step_name}")
        try:
            success = step_func()
            if success:
                print(f"✅ {step_name} completed")
            else:
                print(f"❌ {step_name} failed")
                break
        except Exception as e:
            print(f"❌ {step_name} error: {e}")
            break
    else:
        print("\n🎉 All steps completed successfully!")


def main():
    """Run all examples."""
    print("ImageNet-1k Dataset Downloader Examples")
    print("=" * 50)
    
    # Note: These examples are for demonstration
    # In practice, you would run the actual download
    print("Note: These examples demonstrate usage patterns.")
    print("For actual download, use the command line interface:")
    print("python data_transformer.py --output-dir /path/to/imagenet")
    print()
    
    # Example 1: Basic download (commented out to avoid actual download)
    # example_basic_download()
    
    # Example 2: Resume download
    example_resume_download()
    
    # Example 3: Validate dataset
    example_validate_only()
    
    # Example 4: Custom credentials
    example_custom_credentials()
    
    # Example 5: Step-by-step
    example_step_by_step()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo actually download ImageNet dataset:")
    print("1. Setup Kaggle API: python setup_kaggle.py --instructions")
    print("2. Download dataset: python data_transformer.py --output-dir /path/to/imagenet")


if __name__ == "__main__":
    main()


