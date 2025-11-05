# HuggingFace Model Cache Directory Guide

This guide explains how to configure where HuggingFace models are downloaded and cached.

## Overview

By default, HuggingFace models are downloaded to `~/.cache/huggingface/` (or `C:\Users\<username>\.cache\huggingface\` on Windows). You can configure a custom location using environment variables or code parameters.

## Methods to Set Cache Directory

### Method 1: Environment Variable (Recommended)

Set one of these environment variables before running your scripts:

#### Option A: Custom Variable (HF_MODELS_CACHE)
```bash
# Linux/Mac
export HF_MODELS_CACHE=/path/to/your/models

# Windows (PowerShell)
$env:HF_MODELS_CACHE="C:\path\to\your\models"

# Windows (Command Prompt)
set HF_MODELS_CACHE=C:\path\to\your\models
```

#### Option B: Standard HuggingFace Variable (HF_HOME)
```bash
# Linux/Mac
export HF_HOME=/path/to/huggingface

# Note: With HF_HOME, models are stored in HF_HOME/hub/
# Windows (PowerShell)
$env:HF_HOME="C:\path\to\huggingface"

# Windows (Command Prompt)
set HF_HOME=C:\path\to\huggingface
```

#### Option C: Transformers Variable (TRANSFORMERS_CACHE)
```bash
# Linux/Mac
export TRANSFORMERS_CACHE=/path/to/your/models

# Windows (PowerShell)
$env:TRANSFORMERS_CACHE="C:\path\to\your\models"

# Windows (Command Prompt)
set TRANSFORMERS_CACHE=C:\path\to\your\models
```

**Priority Order:**
1. `HF_MODELS_CACHE` (highest priority - custom variable)
2. `HF_HOME` (standard HuggingFace variable)
3. `TRANSFORMERS_CACHE` (standard transformers variable)
4. Default location `~/.cache/huggingface/` (lowest priority)

### Method 2: Permanent Environment Variable (Windows)

**Windows (User-level):**
1. Open System Properties → Environment Variables
2. Click "New" under User variables
3. Variable name: `HF_MODELS_CACHE`
4. Variable value: `C:\path\to\your\models`
5. Click OK and restart your terminal

**Windows (System-level):**
```powershell
# Run PowerShell as Administrator
[System.Environment]::SetEnvironmentVariable('HF_MODELS_CACHE', 'C:\path\to\your\models', 'Machine')
```

### Method 3: Permanent Environment Variable (Linux/Mac)

Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export HF_MODELS_CACHE=/path/to/your/models
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Method 4: In Code (Advanced)

You can also pass `cache_dir` directly when initializing models (though this requires code changes):

```python
from models.huggingface_models import HuggingfaceModel

model = HuggingfaceModel(
    model_name="Llama-3.2-1B",
    max_new_tokens=100,
    cache_dir="/path/to/your/models"
)
```

## Use Cases

### 1. Shared Network Storage (Cluster)

If you're on a cluster with shared storage, you can set the cache to a shared location:

```bash
# Example: Shared storage on cluster
export HF_MODELS_CACHE=/shared/models/huggingface

# Or if using SLURM, set in your job script:
#SBATCH --export=HF_MODELS_CACHE=/shared/models/huggingface
```

**Benefits:**
- Models downloaded once, accessible to all users
- Saves disk space
- Faster for subsequent users (no re-download)

### 2. Large Local Disk

If you have a large local disk (e.g., for 70B models):

```bash
# Example: Large local disk
export HF_MODELS_CACHE=/local/ssd/models/huggingface
```

**Benefits:**
- Faster access than network storage
- Can store large models locally

### 3. Specific Project Directory

Organize models by project:

```bash
# Example: Project-specific cache
export HF_MODELS_CACHE=/home/user/projects/nllSAR/models
```

### 4. Cloud Storage (Sync)

If you use cloud storage sync:

```bash
# Example: Synced cloud storage
export HF_MODELS_CACHE=/home/user/Dropbox/models/huggingface
```

## Verification

### Check Current Cache Location

```python
import os
from models.huggingface_models import get_hf_cache_dir

cache_dir = get_hf_cache_dir()
print(f"Current cache directory: {cache_dir}")
```

### Check Logs

When you run your scripts, you'll see log messages like:
```
INFO: Using HuggingFace model cache directory: /path/to/your/models
```

Or if using default:
```
INFO: Using default HuggingFace cache directory (check HF_HOME or set HF_MODELS_CACHE)
```

### Verify Model Location

After downloading a model, check where it's stored:

```python
from transformers import AutoModelForCausalLM
import os

# Check where a model would be cached
cache_dir = os.getenv('HF_MODELS_CACHE') or os.getenv('HF_HOME')
if cache_dir:
    if os.getenv('HF_HOME') and not os.getenv('HF_MODELS_CACHE'):
        cache_dir = os.path.join(cache_dir, 'hub')
    print(f"Models will be in: {cache_dir}")
    print(f"Example model path: {cache_dir}/models--meta-llama--Llama-3.2-1B")
```

## Examples

### Example 1: Linux Cluster with Shared Storage

```bash
# In your SLURM script or before running
export HF_MODELS_CACHE=/shared/scratch/models/huggingface

# Create directory if it doesn't exist
mkdir -p $HF_MODELS_CACHE

# Run your script
python run_greedy_decoding.py
```

### Example 2: Windows with Large D Drive

```powershell
# Set environment variable
$env:HF_MODELS_CACHE="D:\Models\HuggingFace"

# Create directory if needed
New-Item -ItemType Directory -Force -Path $env:HF_MODELS_CACHE

# Run your script
python run_greedy_decoding.py
```

### Example 3: Mac with External Drive

```bash
# Set cache to external drive
export HF_MODELS_CACHE=/Volumes/External/models/huggingface

# Create directory
mkdir -p $HF_MODELS_CACHE

# Run script
python run_greedy_decoding.py
```

## Important Notes

### Model Storage Structure

Models are stored with a specific structure:
```
cache_dir/
├── models--meta-llama--Llama-3.2-1B/
│   ├── snapshots/
│   │   └── <hash>/
│   │       ├── config.json
│   │       ├── model files...
│   └── refs/
├── models--meta-llama--Llama-3.1-70B-Instruct/
│   └── ...
```

### Disk Space Requirements

Large models require significant space:
- **Llama-3.2-1B**: ~2 GB
- **Llama-3-8B**: ~16 GB
- **Llama-3.1-70B**: ~140 GB (FP16) or ~70 GB (8-bit)

Make sure your cache directory has enough space!

### Multiple Users (Cluster)

If multiple users share the same cache directory:
- First user downloads the model
- Subsequent users use the cached version
- Make sure permissions allow all users to read the cache

### Moving Existing Cache

If you want to move an existing cache:

```bash
# 1. Stop all running processes using models
# 2. Copy the cache directory
cp -r ~/.cache/huggingface /new/location/huggingface

# 3. Set environment variable
export HF_MODELS_CACHE=/new/location/huggingface

# 4. Verify
python -c "from models.huggingface_models import get_hf_cache_dir; print(get_hf_cache_dir())"
```

## Troubleshooting

### Issue: Models Still Downloading to Default Location

**Solution:**
1. Check environment variable is set: `echo $HF_MODELS_CACHE`
2. Make sure you set it before running Python
3. Restart your terminal/IDE after setting permanent variables

### Issue: Permission Denied

**Solution:**
```bash
# Make sure you have write permissions
chmod -R u+w /path/to/cache

# Or create directory with proper permissions
mkdir -p /path/to/cache
chmod 755 /path/to/cache
```

### Issue: Not Enough Space

**Solution:**
- Check available space: `df -h /path/to/cache`
- Use a different location with more space
- Clean up old models: `huggingface-cli scan-cache`

### Issue: Models Not Found After Moving

**Solution:**
- Models are referenced by hash, so moving should work
- If issues persist, re-download (will use new cache location)

## Summary

✅ **Set `HF_MODELS_CACHE` environment variable** to control where models are stored
✅ **Works with all models** (Llama, Mistral, Falcon, etc.)
✅ **Automatic** - no code changes needed
✅ **Cluster-friendly** - can share cache across users
✅ **Logs show** which cache directory is being used

The code automatically detects and uses your custom cache directory!

