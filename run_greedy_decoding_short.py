#!/usr/bin/env python3
"""
Script to run answer generation with greedy decoding and short answer prompts.
This generates single, most probable short answers (brief phrases, not full sentences)
suitable for LLM-as-judge evaluation.
"""

import subprocess
import sys
import os
from datetime import datetime

def main():
    # Change to the src directory
    os.chdir('src')
    
    # Generate unique experiment name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'greedy_short_{timestamp}'
    
    # Default arguments for greedy decoding with short answers
    # Short prompt asks for brief, concise answers (few words/phrases, not full sentences)
    cmd = [
        'python', 'generate_answers.py',
        '--model_name', 'Llama-3.2-1B',  # Change this to your preferred model
        '--dataset', 'trivia_qa',  # Change this to your preferred dataset
        '--num_samples', '100',  # Start with a small number for testing
        '--num_few_shot', '5',
        '--temperature', '0.0',  # Greedy decoding (temperature=0)
        '--num_generations', '1',  # Only one generation per question (greedy)
        '--brief_prompt', 'short',  # Short answer prompt (brief phrases, not sentences)
        '--enable_brief',  # Enable the brief prompt (boolean flag)
        '--brief_always',  # Always use brief prompt (boolean flag)
        '--no-compute_uncertainties',  # Skip uncertainty computation for now
        '--no-compute_p_true',  # Skip p_true computation (avoids token limit issues with long contexts)
        '--no-get_training_set_generations',  # Skip training set for now
        '--use_context',  # Use context for better answers (boolean flag)
        # Metric: Use 'squad' for generation (no model needed, saves memory)
        # LLM judge metrics can be computed later using compute_uncertainty_measures.py with --recompute_accuracy
        # Judge options: 'squad', 'llm_gpt-4', 'llm_gpt-3.5', 'llm_llama-3.1-70b', 'llm_llama-3-8b', etc.
        #'--metric', 'squad',  # Using squad metric (no model) to avoid memory issues during generation
        '--entity', 'nikosteam',  # WandB entity
        '--project', 'super_guacamole',  # WandB project
        '--experiment_lot', experiment_name  # Unique experiment name
    ]
    
    print("Running command:", ' '.join(cmd))
    print(f"This will generate single, short answers using greedy decoding...")
    print(f"Experiment name: {experiment_name}")
    print(f"Prompt type: short (brief phrases, not full sentences)")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Generation completed successfully!")
        print("Check the wandb logs and generated pickle files for results.")
        print()
        print("To recompute accuracy with an LLM judge later, run:")
        print("  python recompute_accuracy_with_judge.py <wandb_run_id> <metric>")
        print("  Example: python recompute_accuracy_with_judge.py <run_id> llm_llama-3-8b")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running generation: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()

