#!/usr/bin/env python3
"""
Script to run answer generation with greedy decoding and detailed sentence prompts.
This generates single, most probable answers suitable for LLM-as-judge evaluation.
"""

import subprocess
import sys
import os

def main():
    # Change to the src directory
    os.chdir('src')
    
    # Default arguments for greedy decoding with detailed prompts
    cmd = [
        'python', 'generate_answers.py',
        '--model_name', 'Llama-2-7b-chat',  # Change this to your preferred model
        '--dataset', 'squad',  # Change this to your preferred dataset
        '--num_samples', '100',  # Start with a small number for testing
        '--num_few_shot', '5',
        '--temperature', '0.0',  # Greedy decoding
        '--num_generations', '1',  # Only one generation per question
        '--brief_prompt', 'detailed',  # Use the new detailed prompt
        '--enable_brief', 'True',
        '--brief_always', 'True',
        '--compute_uncertainties', 'False',  # Skip uncertainty computation for now
        '--get_training_set_generations', 'False',  # Skip training set for now
        '--use_context', 'True',  # Use context for better answers
        '--metric', 'llm_gpt-4',  # Use GPT-4 as judge
        '--experiment_lot', 'greedy_decoding_detailed_answers'
    ]
    
    print("Running command:", ' '.join(cmd))
    print("This will generate single, detailed answers using greedy decoding...")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Generation completed successfully!")
        print("Check the wandb logs and generated pickle files for results.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running generation: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()
