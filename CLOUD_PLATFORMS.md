# Cloud GPU Platforms for Running Experiments

This guide covers platforms where you can run the experiments, especially for large models like Llama-3.1-70B.

## Cloud GPU Platforms

### 1. **RunPod** (Recommended for Cost-Effective)
- **GPUs Available:** A100 40GB, A100 80GB, H100 80GB
- **Pricing:** 
  - A100 40GB: ~$1.00-1.50/hour
  - A100 80GB: ~$2.00-3.00/hour
  - H100 80GB: ~$4.00-6.00/hour
- **Pros:** 
  - Very affordable
  - Persistent storage available
  - Easy setup with Docker
  - Community templates available
- **Cons:** 
  - Less enterprise support
  - May have availability issues during peak times
- **Website:** https://www.runpod.io
- **Best for:** Budget-conscious users, personal projects

### 2. **Vast.ai** (Budget Option)
- **GPUs Available:** Various GPUs including A100, RTX 4090, H100
- **Pricing:** 
  - A100 40GB: ~$0.80-1.20/hour (spot pricing)
  - A100 80GB: ~$1.50-2.50/hour
- **Pros:** 
  - Very cheap (community marketplace)
  - Good for short experiments
- **Cons:** 
  - Less reliable (community GPUs)
  - No guaranteed availability
  - May require technical setup
- **Website:** https://vast.ai
- **Best for:** Short experiments, budget users

### 3. **Lambda Labs** (Developer-Friendly)
- **GPUs Available:** A100 40GB, A100 80GB, H100
- **Pricing:** 
  - A100 40GB: ~$1.10/hour
  - A100 80GB: ~$2.50/hour
  - H100: ~$8.00/hour
- **Pros:** 
  - Developer-friendly
  - Good documentation
  - Pre-configured environments
  - Reliable infrastructure
- **Cons:** 
  - Slightly more expensive than RunPod
- **Website:** https://lambdalabs.com
- **Best for:** Developers, ML practitioners

### 4. **Google Cloud Platform (GCP)**
- **GPUs Available:** A100, H100, T4, V100
- **Pricing:** 
  - A100 40GB: ~$2.50-3.00/hour
  - A100 80GB: ~$4.00-5.00/hour
  - H100: ~$8.00-10.00/hour
- **Pros:** 
  - Enterprise-grade reliability
  - Excellent for long-term projects
  - Good integration with other GCP services
  - Preemptible instances (cheaper)
- **Cons:** 
  - More expensive than smaller providers
  - Requires credit card setup
- **Website:** https://cloud.google.com
- **Best for:** Enterprise, long-term projects

### 5. **Amazon Web Services (AWS)**
- **GPUs Available:** A100, H100, T4, V100
- **Pricing:** 
  - A100 40GB: ~$3.00-4.00/hour
  - A100 80GB: ~$5.00-6.00/hour
  - H100: ~$10.00-12.00/hour
- **Pros:** 
  - Most comprehensive cloud platform
  - Spot instances (up to 90% discount)
  - Excellent for production workloads
- **Cons:** 
  - Most expensive option
  - Complex pricing structure
- **Website:** https://aws.amazon.com
- **Best for:** Enterprise, production deployments

### 6. **Azure (Microsoft)**
- **GPUs Available:** A100, H100, V100
- **Pricing:** Similar to AWS
- **Pros:** 
  - Enterprise integration
  - Good for Microsoft ecosystem
- **Cons:** 
  - Expensive
  - Complex setup
- **Website:** https://azure.microsoft.com

### 7. **Paperspace Gradient**
- **GPUs Available:** A100, RTX 5000, V100
- **Pricing:** 
  - A100 40GB: ~$2.00-3.00/hour
- **Pros:** 
  - ML-focused platform
  - Good notebook integration
  - Easy to use
- **Cons:** 
  - Limited GPU selection
- **Website:** https://www.paperspace.com

### 8. **CoreWeave** (Specialized for AI)
- **GPUs Available:** A100, H100, RTX 4090
- **Pricing:** Competitive with RunPod
- **Pros:** 
  - AI/ML specialized
  - Good performance
  - Fast provisioning
- **Cons:** 
  - Smaller provider
- **Website:** https://www.coreweave.com

## HPC Clusters (University/Research)

The codebase has SLURM support, so it can run on HPC clusters:

### Features:
- **SLURM integration:** Code detects `SLURM_JOB_ID` environment variable
- **Shared storage:** Uses `SCRATCH_DIR` environment variable
- **No cost:** Usually free for researchers/students

### How to Use:
1. Request access to your institution's HPC cluster
2. Submit jobs via SLURM:
   ```bash
   sbatch --gres=gpu:a100:2 run_greedy_decoding.sh
   ```
3. Code automatically detects SLURM environment

## Recommended Platforms by Use Case

### For Quick Experiments (1-2 hours):
- **RunPod** or **Vast.ai** (cheapest)
- **Cost:** ~$2-5 for a quick run

### For Regular Experiments:
- **Lambda Labs** or **RunPod** (balanced cost/reliability)
- **Cost:** ~$5-15 per experiment run

### For Production/Research:
- **Google Cloud** or **AWS** (reliable, enterprise-grade)
- **Cost:** ~$10-30 per experiment run
- Use spot/preemptible instances for 50-90% savings

### For Long-Term Projects:
- **RunPod** with persistent storage (most cost-effective)
- **Google Cloud** or **AWS** (if you need enterprise features)

## Setup Instructions

### RunPod Setup:
1. Create account at https://www.runpod.io
2. Create a GPU pod (A100 80GB recommended)
3. Select PyTorch template
4. SSH into the pod
5. Clone your repository
6. Install dependencies
7. Run experiments

### Lambda Labs Setup:
1. Create account at https://lambdalabs.com
2. Launch instance (A100 80GB)
3. SSH into instance
4. Clone repository
5. Run experiments

### Google Cloud Setup:
1. Create GCP account
2. Enable GPU quota
3. Create VM with GPU:
   ```bash
   gcloud compute instances create gpu-instance \
     --machine-type=n1-standard-4 \
     --accelerator="type=nvidia-tesla-a100,count=1" \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --zone=us-central1-a
   ```
4. Install NVIDIA drivers and CUDA
5. Clone repository and run

## Cost Estimates

### Running 1000 Examples:
- **Generation (Llama-3.2-1B):** ~10-20 minutes
- **Judge Evaluation (Llama-3-70B, 8-bit):** ~30-60 minutes
- **Total time:** ~1-2 hours

**Cost per experiment:**
- RunPod: $2-6
- Lambda Labs: $2.50-7.50
- Google Cloud: $5-10
- AWS: $6-12

### Monthly Usage (10 experiments):
- RunPod: $20-60/month
- Lambda Labs: $25-75/month
- Google Cloud: $50-100/month
- AWS: $60-120/month

## Tips for Cost Optimization

1. **Use Spot/Preemptible Instances:**
   - 50-90% cheaper
   - May be interrupted (save checkpoints)

2. **Sequential Workflow:**
   - Generate first, then evaluate
   - Only load one model at a time
   - Saves memory and allows smaller instances

3. **Batch Processing:**
   - Process multiple experiments in one session
   - Reduces startup overhead

4. **Use Persistent Storage:**
   - Keep data between sessions
   - Avoid re-downloading models

5. **Monitor Usage:**
   - Set up billing alerts
   - Stop instances when not in use

## Quick Start Commands

### RunPod (after SSH):
```bash
# Clone repository
git clone <your-repo>
cd nllSAR

# Install dependencies
pip install -r requirements.txt

# Run generation
python run_greedy_decoding.py

# Run judge evaluation (after generation)
python recompute_accuracy_with_judge.py <wandb_run_id> llm_llama-3-70b
```

### Lambda Labs (after SSH):
```bash
# Same as RunPod
```

## Recommended Setup for Your Use Case

**For running with 70B models:**
- **Platform:** RunPod or Lambda Labs
- **Instance:** 1× A100 80GB (or 2× A100 40GB)
- **Storage:** 100GB+ for models and data
- **Estimated cost:** $2-5 per experiment

**For budget-conscious:**
- **Platform:** Vast.ai (spot pricing)
- **Instance:** 1× A100 80GB
- **Estimated cost:** $1-3 per experiment

**For reliability:**
- **Platform:** Google Cloud or AWS
- **Instance:** 1× A100 80GB
- **Use preemptible/spot instances**
- **Estimated cost:** $3-6 per experiment


