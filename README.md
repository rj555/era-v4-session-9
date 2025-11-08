# ImageNet-1k Dataset Downloader

A comprehensive Python script to download and organize the ImageNet-1k dataset (~170GB) for ResNet50 training on EC2 instances using the Kaggle API.

## Features

- ✅ **Kaggle API Integration**: Secure download using your Kaggle credentials
- ✅ **Progress Tracking**: Resume interrupted downloads
- ✅ **Dataset Validation**: Comprehensive integrity checks
- ✅ **Automatic Organization**: Structures data for ResNet50 training
- ✅ **Error Handling**: Robust error recovery and logging
- ✅ **Space Management**: Disk space checking and cleanup options
- ✅ **Logging**: Detailed progress and error logging

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Kaggle API

```bash
# Get your Kaggle API key from https://www.kaggle.com/account
# Download kaggle.json and place it in ~/.kaggle/

# Setup credentials
python setup_kaggle.py --setup-credentials /path/to/kaggle.json

# Validate setup
python setup_kaggle.py --validate
```

### 3. Download ImageNet Dataset

```bash
# Basic download
python data_transformer.py --output-dir /path/to/imagenet

# Resume interrupted download
python data_transformer.py --output-dir /path/to/imagenet --resume

# Keep raw files (don't cleanup)
python data_transformer.py --output-dir /path/to/imagenet --no-cleanup
```

## Usage

### Command Line Options

```bash
python data_transformer.py [OPTIONS]

Options:
  --output-dir PATH          Directory to store the dataset (required)
  --kaggle-credentials PATH  Path to kaggle.json file (optional)
  --resume                   Resume interrupted download
  --no-cleanup              Don't clean up raw files after processing
  --validate-only           Only validate existing dataset
```

### Example Commands

```bash
# Full download with resume capability
python data_transformer.py --output-dir /data/imagenet --resume

# Download with custom Kaggle credentials
python data_transformer.py --output-dir /data/imagenet --kaggle-credentials /path/to/kaggle.json

# Validate existing dataset
python data_transformer.py --output-dir /data/imagenet --validate-only
```

## Dataset Structure

After successful download, the dataset will be organized as:

```
/path/to/imagenet/
├── processed/
│   ├── train/
│   │   ├── n01440764/     # Class directories
│   │   │   ├── image1.JPEG
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       ├── n01440764/
│       └── ...
├── raw/                   # Raw downloaded files (cleaned up by default)
├── download_progress.json
└── download.log
```

## Requirements

### System Requirements
- **Disk Space**: ~200GB (170GB dataset + 30GB buffer)
- **RAM**: 8GB+ recommended
- **Internet**: Stable connection for large download
- **OS**: Linux/macOS (tested on Ubuntu 20.04+)

### Python Requirements
- Python 3.7+
- Kaggle API (`pip install kaggle`)
- Standard library modules (no additional dependencies)

## Kaggle API Setup

### Method 1: Manual Setup
1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json`
5. Place in `~/.kaggle/kaggle.json`
6. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Method 2: Using Setup Script
```bash
# Show instructions
python setup_kaggle.py --instructions

# Setup from existing kaggle.json
python setup_kaggle.py --setup-credentials /path/to/kaggle.json

# Validate setup
python setup_kaggle.py --validate
```

## Troubleshooting

### Common Issues

**1. Kaggle API Authentication Failed**
```bash
# Check credentials
python setup_kaggle.py --validate

# Re-setup if needed
python setup_kaggle.py --setup-credentials /path/to/kaggle.json
```

**2. Insufficient Disk Space**
- The script checks disk space and warns if insufficient
- Ensure at least 200GB free space
- Use `--no-cleanup` to keep raw files if needed

**3. Download Interruption**
```bash
# Resume download
python data_transformer.py --output-dir /path/to/imagenet --resume
```

**4. Dataset Validation Failed**
```bash
# Check what went wrong
tail -f /path/to/imagenet/download.log

# Re-validate
python data_transformer.py --output-dir /path/to/imagenet --validate-only
```

### Log Files

- **download.log**: Detailed progress and error logs
- **download_progress.json**: Resume information
- **Console output**: Real-time progress updates

## Performance Tips

### For EC2 Instances
- Use **c5.2xlarge** or larger for good network performance
- Consider **c5.4xlarge** for faster processing
- Use **gp3** EBS volumes for better I/O performance
- Enable **Enhanced Networking** for better bandwidth

### Network Optimization
- Run during off-peak hours for better download speeds
- Use `--resume` if connection is unstable
- Consider using a VPN if experiencing slow speeds

## Integration with ResNet50 Training

The organized dataset is ready for PyTorch/TensorFlow training:

```python
# PyTorch example
from torchvision import datasets, transforms

# Data loading
train_dataset = datasets.ImageFolder(
    '/path/to/imagenet/processed/train',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
)

val_dataset = datasets.ImageFolder(
    '/path/to/imagenet/processed/val',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the log files for error details
3. Open an issue with detailed error information