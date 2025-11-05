# Environment Setup Guide

This guide explains how to set up the conda environment for the nllSAR project.

## Quick Start

### Create Environment from YAML

```bash
# Create the environment
conda env create -f nllSAR.yml

# Activate the environment
conda activate nllSAR
```

### Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10, 3.11, or 3.12

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Check key packages
python -c "import transformers; import accelerate; import wandb; print('All packages installed successfully!')"
```

## Customizing CUDA Version

The environment file uses `pytorch-cuda>=11.8` by default. If you need a specific CUDA version:

### For CUDA 11.8:
```bash
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### For CUDA 12.1:
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### For CUDA 12.4:
```bash
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

## Updating the Environment

### Update all packages:
```bash
conda activate nllSAR
conda update --all
pip install --upgrade pip
pip list --outdated | cut -d ' ' -f1 | xargs -n1 pip install -U
```

### Update specific packages:
```bash
conda activate nllSAR
pip install --upgrade transformers accelerate wandb
```

## Export Current Environment

If you make changes and want to update the YAML file:

```bash
conda activate nllSAR
conda env export > nllSAR.yml
```

**Note:** The exported file will include your system-specific paths. You may want to manually clean it up to make it portable.

## Platform-Specific Notes

### Windows
- The environment should work on Windows
- Make sure you have Visual C++ Redistributable installed
- CUDA support requires NVIDIA drivers and CUDA toolkit

### Linux
- Works best on Linux (especially for GPU support)
- CUDA toolkit installation may be required separately

### macOS
- PyTorch with CUDA is not available on macOS
- Use CPU-only version: `conda install pytorch -c pytorch` (remove pytorch-cuda)

## Troubleshooting

### CUDA Not Available
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with CUDA
conda activate nllSAR
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia --force-reinstall
```

### Package Conflicts
```bash
# Create fresh environment
conda env remove -n nllSAR
conda env create -f nllSAR.yml
```

### HuggingFace Authentication
```bash
# Login to HuggingFace for gated models (Llama, etc.)
huggingface-cli login
# Or set environment variable:
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

### Memory Issues with Large Models
- Make sure you have enough RAM/VRAM
- Consider using quantization (8-bit or 4-bit) for 70B models
- See `GPU_REQUIREMENTS.md` for details

## Virtual Environments Alternative

If you prefer `venv` or `virtualenv` instead of conda:

```bash
# Create virtual environment
python -m venv nllSAR_env

# Activate (Linux/Mac)
source nllSAR_env/bin/activate

# Activate (Windows)
nllSAR_env\Scripts\activate

# Install packages
pip install -r requirements.txt  # If you create one
# Or install manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate huggingface-hub datasets evaluate wandb openai rouge-score nltk tiktoken bitsandbytes
```

## Cloud GPU Setup

When using cloud platforms (RunPod, Lambda Labs, etc.):

1. **SSH into the instance**
2. **Install Miniconda:**
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   source ~/.bashrc
   ```

3. **Create environment:**
   ```bash
   conda env create -f nllSAR.yml
   conda activate nllSAR
   ```

4. **Verify GPU:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

## Next Steps

After setting up the environment:

1. **Clone the repository** (if not already done)
2. **Set up WandB:**
   ```bash
   wandb login
   ```

3. **Set up HuggingFace:**
   ```bash
   huggingface-cli login
   ```

4. **Run a test:**
   ```bash
   cd src
   python generate_answers.py --help
   ```


