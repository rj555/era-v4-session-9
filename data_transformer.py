#!/usr/bin/env python3
"""
ImageNet-1k Dataset Downloader and Transformer

This script downloads the ImageNet-1k dataset (~170GB) using the Kaggle API,
extracts and organizes it for ResNet50 training. It includes progress tracking,
error handling, resume capability, and dataset validation.

Requirements:
- Kaggle API key configured in ~/.kaggle/kaggle.json
- Sufficient disk space (~200GB recommended)
- Stable internet connection

Usage:
    python data_transformer.py --output-dir /path/to/imagenet --resume
"""

import os
import sys
import json
import shutil
import logging
import argparse
import hashlib
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import subprocess
from threading import Thread, Event

try:
    # Optional dependency for progress bars
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Error: Kaggle API not installed. Run: pip install kaggle")
    sys.exit(1)


class ImageNetDownloader:
    """ImageNet-1k dataset downloader and transformer."""
    
    def __init__(self, output_dir: str, kaggle_credentials_path: str = None):
        """
        Initialize the ImageNet downloader.
        
        Args:
            output_dir: Directory to store the dataset
            kaggle_credentials_path: Path to kaggle.json (optional)
        """
        self.output_dir = Path(output_dir)
        self.kaggle_credentials_path = kaggle_credentials_path
        self.logger = self._setup_logging()
        
        # Dataset configuration
        self.dataset_name = "imagenet-object-localization-challenge"
        self.expected_size_gb = 170
        self.expected_classes = 1000
        
        # Progress tracking
        self.progress_file = self.output_dir / "download_progress.json"
        self.downloaded_files = set()
        
        # Setup directories
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "val"
        
        self._setup_directories()
        self._setup_kaggle_api()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('ImageNetDownloader')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.output_dir / "download.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_directories(self):
        """Create necessary directories."""
        for directory in [self.output_dir, self.raw_dir, self.processed_dir, 
                         self.train_dir, self.val_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def _setup_kaggle_api(self):
        """Setup Kaggle API with credentials."""
        try:
            # Set up Kaggle API
            if self.kaggle_credentials_path:
                os.environ['KAGGLE_CONFIG_DIR'] = str(Path(self.kaggle_credentials_path).parent)
            
            self.api = KaggleApi()
            self.api.authenticate()
            self.logger.info("Kaggle API authenticated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to authenticate with Kaggle API: {e}")
            self.logger.error("Please ensure kaggle.json is in ~/.kaggle/ or provide path via --kaggle-credentials")
            raise
    
    def _load_progress(self) -> Dict:
        """Load download progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.downloaded_files = set(progress.get('downloaded_files', []))
                    self.logger.info(f"Loaded progress: {len(self.downloaded_files)} files already downloaded")
                    return progress
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}")
        return {}
    
    def _save_progress(self, progress: Dict):
        """Save download progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
    
    def _check_disk_space(self) -> bool:
        """Check if there's enough disk space."""
        try:
            statvfs = os.statvfs(self.output_dir)
            free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            
            self.logger.info(f"Available disk space: {free_space_gb:.2f} GB")
            
            if free_space_gb < self.expected_size_gb * 1.2:  # 20% buffer
                self.logger.warning(f"Low disk space: {free_space_gb:.2f} GB available, "
                                  f"{self.expected_size_gb * 1.2:.2f} GB recommended")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Could not check disk space: {e}")
            return True  # Continue anyway
    
    def download_dataset(self, resume: bool = True) -> bool:
        """
        Download the ImageNet dataset from Kaggle.
        
        Args:
            resume: Whether to resume interrupted downloads
            
        Returns:
            True if download successful, False otherwise
        """
        self.logger.info("Starting ImageNet-1k dataset download...")
        
        # Check disk space
        if not self._check_disk_space():
            response = input("Continue with low disk space? (y/N): ")
            if response.lower() != 'y':
                return False
        
        # Load progress if resuming
        if resume:
            self._load_progress()
        
        try:
            # Preflight: verify access to competition files (avoids silent 0% stalls)
            kaggle_cli = shutil.which("kaggle")
            if kaggle_cli is not None:
                preflight = subprocess.run(
                    [kaggle_cli, "competitions", "files", "-c", self.dataset_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                if preflight.returncode != 0:
                    self.logger.error("Kaggle access preflight failed. Common causes: not joined or rules not accepted.")
                    self.logger.error("Action: Open https://www.kaggle.com/competitions/imagenet-object-localization-challenge and click Join/Accept Rules.")
                    for line in (preflight.stdout or "").splitlines():
                        if line.strip():
                            self.logger.error(line)
                    return False
            # Prefer kaggle CLI to allow progress monitoring; fallback to API
            if kaggle_cli is not None:
                self.logger.info(f"Downloading dataset via kaggle CLI: {self.dataset_name}")
                cmd = [
                    kaggle_cli,
                    "competitions",
                    "download",
                    "-c",
                    self.dataset_name,
                    "-p",
                    str(self.raw_dir)
                ]
                self.logger.debug(f"Running: {' '.join(cmd)}")
                # Start download subprocess
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

                # Start progress monitor thread
                stop_event = Event()
                monitor_thread = Thread(target=self._monitor_download_progress, args=(stop_event,))
                monitor_thread.daemon = True
                monitor_thread.start()

                # Stream output to logger
                try:
                    for line in proc.stdout:  # type: ignore[attr-defined]
                        line = line.rstrip()
                        if line:
                            self.logger.info(line)
                finally:
                    proc.wait()
                    stop_event.set()
                    monitor_thread.join(timeout=5)

                if proc.returncode != 0:
                    self.logger.error(f"kaggle CLI exited with code {proc.returncode}")
                    return False

                self.logger.info("Dataset download completed successfully (kaggle CLI)")
                return True
            else:
                self.logger.info("kaggle CLI not found; falling back to Kaggle API (no live progress)")
                self.logger.info(f"Downloading dataset: {self.dataset_name}")
                self.api.competition_download_files(
                    self.dataset_name,
                    path=str(self.raw_dir),
                    quiet=False
                )
                self.logger.info("Dataset download completed successfully (Kaggle API)")
                return True

        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            return False

    def _monitor_download_progress(self, stop_event: Event):
        """Monitor size of the downloading .zip and render a progress bar/logs."""
        expected_total_bytes = int(self.expected_size_gb * (1024 ** 3))
        last_size = 0
        start_time = time.time()
        bar = None
        last_growth_time = time.time()
        try:
            if tqdm is not None:
                bar = tqdm(
                    total=expected_total_bytes,
                    unit='B',
                    unit_scale=True,
                    desc='Downloading ImageNet',
                    dynamic_ncols=True
                )
                bar.update(0)
            while not stop_event.is_set():
                # Find the largest growing zip file in raw_dir
                candidates = []
                # include common temp/partial extensions as created by kaggle/requests
                for pattern in ["*.zip", "*.part", "*.tmp", "*.download"]:
                    candidates.extend(self.raw_dir.glob(pattern))
                current_size = 0
                if candidates:
                    current_size = max(p.stat().st_size for p in candidates)
                delta = max(0, current_size - last_size)
                last_size = current_size
                if bar is not None:
                    bar.update(delta)
                    bar.set_postfix_str(f"{current_size/(1024**3):.2f} GB")
                else:
                    # Fallback logging every 5 seconds
                    self.logger.info(f"Downloading... {current_size/(1024**3):.2f} GB")
                # Stall detection: if no growth for 3 minutes, warn prominently
                if delta > 0:
                    last_growth_time = time.time()
                else:
                    if time.time() - last_growth_time > 180:
                        self.logger.warning("Download appears stalled (no size growth for >3 minutes).\n"
                                            "Checklist: 1) Ensure competition rules accepted. 2) Check network. 3) Try again later.\n"
                                            "If using corporate/VPN, disable temporarily. Will continue monitoring...")
                        last_growth_time = time.time()  # avoid spamming
                stop_event.wait(5)
        except Exception as e:
            self.logger.debug(f"Progress monitor error: {e}")
        finally:
            if bar is not None:
                bar.close()
    
    def extract_dataset(self) -> bool:
        """
        Extract downloaded dataset files.
        
        Returns:
            True if extraction successful, False otherwise
        """
        self.logger.info("Extracting dataset files...")
        
        extracted_files = []
        
        try:
            for file_path in self.raw_dir.glob("*.zip"):
                self.logger.info(f"Extracting {file_path.name}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    members = zip_ref.infolist()
                    if tqdm is not None:
                        with tqdm(total=len(members), desc=f"Extract {file_path.name}", unit="files", dynamic_ncols=True) as pbar:
                            for m in members:
                                zip_ref.extract(m, self.raw_dir)
                                pbar.update(1)
                    else:
                        zip_ref.extractall(self.raw_dir)
                    extracted_files.append(file_path.name)
            
            for file_path in self.raw_dir.glob("*.tar.gz"):
                self.logger.info(f"Extracting {file_path.name}...")
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    members = tar_ref.getmembers()
                    if tqdm is not None:
                        with tqdm(total=len(members), desc=f"Extract {file_path.name}", unit="files", dynamic_ncols=True) as pbar:
                            for m in members:
                                tar_ref.extract(m, self.raw_dir)
                                pbar.update(1)
                    else:
                        tar_ref.extractall(self.raw_dir)
                    extracted_files.append(file_path.name)
            
            self.logger.info(f"Extracted {len(extracted_files)} files")
            return True
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return False
    
    def organize_dataset(self) -> bool:
        """
        Organize dataset into train/val structure for ResNet50 training.
        
        Returns:
            True if organization successful, False otherwise
        """
        self.logger.info("Organizing dataset structure...")
        
        try:
            # Find extracted directories
            extracted_dirs = []
            for item in self.raw_dir.iterdir():
                if item.is_dir() and item.name not in ['train', 'val', 'test']:
                    extracted_dirs.append(item)
            
            if not extracted_dirs:
                self.logger.error("No extracted directories found")
                return False
            
            # Look for train and validation data
            for extracted_dir in extracted_dirs:
                self.logger.info(f"Processing directory: {extracted_dir.name}")
                
                # Check if this is a train or val directory
                if 'train' in extracted_dir.name.lower():
                    target_dir = self.train_dir
                elif 'val' in extracted_dir.name.lower() or 'validation' in extracted_dir.name.lower():
                    target_dir = self.val_dir
                else:
                    # Try to determine from structure
                    subdirs = [d for d in extracted_dir.iterdir() if d.is_dir()]
                    if len(subdirs) > 100:  # Likely train data
                        target_dir = self.train_dir
                    else:
                        target_dir = self.val_dir
                
                # Copy class directories
                class_dirs = [d for d in extracted_dir.iterdir() if d.is_dir()]
                self.logger.info(f"Found {len(class_dirs)} class directories")
                
                iterator = class_dirs
                pbar = None
                if tqdm is not None:
                    pbar = tqdm(total=len(class_dirs), desc=f"Copy {extracted_dir.name}", unit="classes", dynamic_ncols=True)
                try:
                    for class_dir in iterator:
                        target_class_dir = target_dir / class_dir.name
                        if not target_class_dir.exists():
                            shutil.copytree(class_dir, target_class_dir)
                            self.logger.debug(f"Copied {class_dir.name} to {target_dir.name}")
                        if pbar is not None:
                            pbar.update(1)
                finally:
                    if pbar is not None:
                        pbar.close()
            
            # Validate structure
            train_classes = len([d for d in self.train_dir.iterdir() if d.is_dir()])
            val_classes = len([d for d in self.val_dir.iterdir() if d.is_dir()])
            
            self.logger.info(f"Dataset organized: {train_classes} train classes, {val_classes} val classes")
            
            if train_classes == 0 or val_classes == 0:
                self.logger.error("Dataset organization failed - no classes found")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset organization failed: {e}")
            return False
    
    def validate_dataset(self) -> bool:
        """
        Validate the downloaded and organized dataset.
        
        Returns:
            True if validation successful, False otherwise
        """
        self.logger.info("Validating dataset...")
        
        try:
            # Check directory structure
            if not self.train_dir.exists() or not self.val_dir.exists():
                self.logger.error("Missing train or val directories")
                return False
            
            # Count classes and images
            train_classes = [d for d in self.train_dir.iterdir() if d.is_dir()]
            val_classes = [d for d in self.val_dir.iterdir() if d.is_dir()]
            
            train_images = sum(len(list(d.glob("*.JPEG"))) + len(list(d.glob("*.jpg"))) 
                             for d in train_classes)
            val_images = sum(len(list(d.glob("*.JPEG"))) + len(list(d.glob("*.jpg"))) 
                           for d in val_classes)
            
            self.logger.info(f"Dataset validation results:")
            self.logger.info(f"  Train classes: {len(train_classes)}")
            self.logger.info(f"  Val classes: {len(val_classes)}")
            self.logger.info(f"  Train images: {train_images}")
            self.logger.info(f"  Val images: {val_images}")
            
            # Basic validation checks
            if len(train_classes) < 100:
                self.logger.error(f"Too few train classes: {len(train_classes)}")
                return False
            
            if len(val_classes) < 100:
                self.logger.error(f"Too few val classes: {len(val_classes)}")
                return False
            
            if train_images < 10000:
                self.logger.error(f"Too few train images: {train_images}")
                return False
            
            if val_images < 1000:
                self.logger.error(f"Too few val images: {val_images}")
                return False
            
            self.logger.info("Dataset validation passed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False
    
    def cleanup_raw_files(self):
        """Clean up raw downloaded files to save space."""
        self.logger.info("Cleaning up raw files...")
        
        try:
            # Remove zip/tar files
            for pattern in ["*.zip", "*.tar.gz", "*.tar"]:
                for file_path in self.raw_dir.glob(pattern):
                    file_path.unlink()
                    self.logger.info(f"Removed {file_path.name}")
            
            # Remove extracted directories if they're not needed
            for item in self.raw_dir.iterdir():
                if item.is_dir() and item.name not in ['train', 'val']:
                    shutil.rmtree(item)
                    self.logger.info(f"Removed directory {item.name}")
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def run_full_pipeline(self, resume: bool = True, cleanup: bool = True) -> bool:
        """
        Run the complete dataset download and transformation pipeline.
        
        Args:
            resume: Whether to resume interrupted downloads
            cleanup: Whether to clean up raw files after processing
            
        Returns:
            True if pipeline successful, False otherwise
        """
        self.logger.info("Starting ImageNet-1k dataset pipeline...")
        
        try:
            # Step 1: Download dataset
            if not self.download_dataset(resume=resume):
                return False
            
            # Step 2: Extract dataset
            if not self.extract_dataset():
                return False
            
            # Step 3: Organize dataset
            if not self.organize_dataset():
                return False
            
            # Step 4: Validate dataset
            if not self.validate_dataset():
                return False
            
            # Step 5: Cleanup (optional)
            if cleanup:
                self.cleanup_raw_files()
            
            self.logger.info("ImageNet-1k dataset pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False


def main():
    """Main function to run the ImageNet downloader."""
    parser = argparse.ArgumentParser(
        description="Download and transform ImageNet-1k dataset for ResNet50 training"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="Directory to store the dataset"
    )
    parser.add_argument(
        "--kaggle-credentials",
        type=str,
        help="Path to kaggle.json file (default: ~/.kaggle/kaggle.json)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted download"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up raw files after processing"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing dataset"
    )
    
    args = parser.parse_args()
    
    # Create downloader instance
    downloader = ImageNetDownloader(
        output_dir=args.output_dir,
        kaggle_credentials_path=args.kaggle_credentials
    )
    
    if args.validate_only:
        # Only validate existing dataset
        success = downloader.validate_dataset()
    else:
        # Run full pipeline
        success = downloader.run_full_pipeline(
            resume=args.resume,
            cleanup=not args.no_cleanup
        )
    
    if success:
        print("✅ ImageNet-1k dataset ready for ResNet50 training!")
        sys.exit(0)
    else:
        print("❌ Dataset preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

