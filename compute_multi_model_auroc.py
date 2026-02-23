#!/usr/bin/env python3
"""
Compute G-NLL and RW-G-NLL AUROC for multi-model family comparison experiments.

This script:
1. Finds all validation_generations.pkl files from experiments
2. Computes G-NLL and RW-G-NLL AUROC for each
3. Aggregates results by model family, model size, and dataset
4. Exports results to JSON and CSV for analysis

Usage:
    python compute_multi_model_auroc.py --wandb_dir src/nikos/uncertainty/wandb --output_dir results/multi_model_auroc
"""

import argparse
import os
import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))

# Import from existing code
from run_gnll_baseline import compute_gnll_auroc


def find_experiment_pickles(wandb_dir, experiment_pattern="multi_model"):
    """Find all validation_generations.pkl files from multi-model experiments."""
    pickles = []
    wandb_path = Path(wandb_dir)
    
    for run_dir in wandb_path.glob("run-*"):
        # Check if this is a multi-model experiment
        files_dir = run_dir / "files"
        if not files_dir.exists():
            continue
        
        # Look for validation_generations.pkl
        pickle_file = files_dir / "validation_generations.pkl"
        if not pickle_file.exists():
            continue
        
        # Try to extract experiment info from run directory
        try:
            # Look for config file
            config_file = files_dir / "config.yaml"
            if config_file.exists():
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                model_name = config.get('model_name', {}).get('value', 'unknown')
                dataset = config.get('dataset', {}).get('value', 'unknown')
                experiment_lot = config.get('experiment_lot', {}).get('value', '')
                
                # Only include multi-model experiments
                if experiment_pattern in experiment_lot.lower():
                    pickles.append({
                        'pickle_path': str(pickle_file),
                        'model_name': model_name,
                        'dataset': dataset,
                        'experiment_lot': experiment_lot,
                        'run_id': run_dir.name.split('-', 1)[1]
                    })
        except Exception as e:
            print(f"Warning: Could not parse config for {run_dir}: {e}")
            continue
    
    return pickles


def parse_model_info(model_name):
    """Parse model family and size from model name."""
    # Normalize model name
    name_lower = model_name.lower()
    
    # Determine family
    if 'llama' in name_lower:
        family = 'Llama'
    elif 'qwen' in name_lower:
        family = 'Qwen'
    elif 'mistral' in name_lower or 'mixtral' in name_lower:
        family = 'Mistral'
    else:
        family = 'Unknown'
    
    # Determine size category
    if '1b' in name_lower or '1.5b' in name_lower or '3b' in name_lower:
        size = 'Small'
    elif '7b' in name_lower or '8b' in name_lower:
        size = 'Large'
    elif '70b' in name_lower or '72b' in name_lower:
        size = 'XLarge'
    else:
        size = 'Unknown'
    
    # Check if quantized
    quantization = None
    if '8bit' in name_lower or '-8bit' in name_lower:
        quantization = '8-bit'
    elif '4bit' in name_lower or '-4bit' in name_lower:
        quantization = '4-bit'
    
    return family, size, quantization


def main():
    parser = argparse.ArgumentParser(description='Compute AUROC for multi-model experiments')
    parser.add_argument('--wandb_dir', type=str, 
                       default='src/nikos/uncertainty/wandb',
                       help='Path to wandb directory')
    parser.add_argument('--output_dir', type=str,
                       default='results/multi_model_auroc',
                       help='Output directory for results')
    parser.add_argument('--experiment_pattern', type=str,
                       default='multi_model',
                       help='Pattern to match experiment names')
    parser.add_argument('--use_rw_gnll', action='store_true',
                       help='Compute RW-G-NLL in addition to G-NLL')
    parser.add_argument('--use_rouge', action='store_true',
                       help='Use ROUGE for correctness (for short answers)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Multi-Model AUROC Computation")
    print("="*80)
    print(f"Wandb directory: {args.wandb_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Experiment pattern: {args.experiment_pattern}")
    print(f"Compute RW-G-NLL: {args.use_rw_gnll}")
    print(f"Use ROUGE: {args.use_rouge}")
    print("="*80)
    print()
    
    # Find experiment pickles
    print("Searching for experiment results...")
    pickles = find_experiment_pickles(args.wandb_dir, args.experiment_pattern)
    print(f"Found {len(pickles)} experiments")
    print()
    
    if len(pickles) == 0:
        print("No experiments found! Make sure to run experiments first.")
        return
    
    # Compute AUROC for each experiment
    results = []
    
    for i, exp in enumerate(pickles, 1):
        print(f"Processing {i}/{len(pickles)}: {exp['model_name']} on {exp['dataset']}")
        
        family, size, quantization = parse_model_info(exp['model_name'])
        
        try:
            auroc_results = compute_gnll_auroc(
                exp['pickle_path'],
                use_rouge=args.use_rouge,
                use_rw_gnll=args.use_rw_gnll,
                model_name=exp['model_name']
            )
            
            if auroc_results:
                result = {
                    'model_name': exp['model_name'],
                    'model_family': family,
                    'model_size': size,
                    'quantization': quantization,
                    'dataset': exp['dataset'],
                    'run_id': exp['run_id'],
                    'experiment_lot': exp['experiment_lot'],
                    **auroc_results
                }
                results.append(result)
                print(f"  ✅ G-NLL AUROC: {auroc_results['G-NLL_AUROC']:.4f}")
                if 'RW-G-NLL_AUROC' in auroc_results:
                    print(f"  ✅ RW-G-NLL AUROC: {auroc_results['RW-G-NLL_AUROC']:.4f}")
            else:
                print(f"  ⚠️  Could not compute AUROC")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    # Save results
    print("="*80)
    print(f"Processed {len(results)} experiments successfully")
    print("="*80)
    print()
    
    if len(results) == 0:
        print("No results to save!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = os.path.join(args.output_dir, 'auroc_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved results to {csv_path}")
    
    # Save to JSON
    json_path = os.path.join(args.output_dir, 'auroc_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to {json_path}")
    
    # Create summary statistics
    print()
    print("="*80)
    print("Summary Statistics")
    print("="*80)
    print()
    
    # Group by model family
    print("By Model Family:")
    family_stats = df.groupby('model_family')['G-NLL_AUROC'].agg(['mean', 'std', 'count'])
    print(family_stats)
    print()
    
    # Group by model size
    print("By Model Size:")
    size_stats = df.groupby('model_size')['G-NLL_AUROC'].agg(['mean', 'std', 'count'])
    print(size_stats)
    print()
    
    # Group by dataset
    print("By Dataset:")
    dataset_stats = df.groupby('dataset')['G-NLL_AUROC'].agg(['mean', 'std', 'count'])
    print(dataset_stats)
    print()
    
    # Best performing models
    print("Top 5 Models by G-NLL AUROC:")
    top_models = df.nlargest(5, 'G-NLL_AUROC')[['model_name', 'dataset', 'G-NLL_AUROC']]
    print(top_models.to_string(index=False))
    print()
    
    print("="*80)
    print("✅ AUROC computation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
