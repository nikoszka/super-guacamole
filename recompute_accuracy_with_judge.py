#!/usr/bin/env python3
"""
Script to recompute accuracy using an LLM judge on already generated answers.
This allows you to generate answers first (without loading the judge model),
then evaluate them later with a judge model.
"""

import subprocess
import sys
import os

def main():
    # Change to the src directory
    os.chdir('src')
    
    # You need to provide the wandb run ID from the generation run
    # You can find this in the wandb URL or output logs
    print("=" * 80)
    print("Recompute Accuracy with LLM Judge")
    print("=" * 80)
    print()
    print("This script will recompute accuracy using an LLM judge on already generated answers.")
    print("You need:")
    print("  1. The wandb run ID from your generation run")
    print("  2. The metric you want to use (e.g., 'llm_llama-3-8b', 'llm_gpt-4')")
    print()
    
    # Get wandb run ID from user or command line
    if len(sys.argv) > 1:
        wandb_runid = sys.argv[1]
    else:
        wandb_runid = input("Enter the wandb run ID from your generation: ").strip()
    
    if len(sys.argv) > 2:
        metric = sys.argv[2]
    else:
        print("\nAvailable metrics:")
        print("  - 'llm_gpt-4' (OpenAI GPT-4)")
        print("  - 'llm_gpt-3.5' (OpenAI GPT-3.5)")
        print("  - 'llm_llama-3-8b' (Llama 3 8B - fits in 8GB GPU)")
        print("  - 'llm_llama-3.1-70b' (Llama 3.1 70B - requires ~35GB GPU)")
        print("  - 'squad' (SQuAD F1 - no model needed)")
        metric = input("Enter the metric to use: ").strip()
    
    # Default arguments for recomputing accuracy
    # Note: compute_uncertainties is only available in 'generate' stage, not 'compute' stage
    cmd = [
        'python', 'compute_uncertainty_measures.py',
        '--eval_wandb_runid', wandb_runid,
        '--metric', metric,
        '--recompute_accuracy',  # This flag triggers recomputation
        '--no-compute_predictive_entropy',  # Skip predictive entropy computation
        '--no-compute_p_ik',  # Skip p_ik computation
        '--no-compute_context_entails_response',  # Skip entailment computation
        '--no-analyze_run',  # Skip analysis for faster recomputation
        '--entity', 'nikosteam',  # WandB entity for new run
        '--project', 'super_guacamole',  # WandB project for new run
        '--restore_entity_eval', 'nikosteam',  # Entity of the run to restore from
    ]
    
    print()
    print("Running command:", ' '.join(cmd))
    print(f"This will recompute accuracy using {metric} as judge...")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Accuracy recomputation completed successfully!")
        print("Check the wandb logs for updated accuracy metrics.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error recomputing accuracy: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Recomputation interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()

