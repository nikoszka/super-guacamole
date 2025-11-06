#!/usr/bin/env python3
"""
Streamlined script to generate short and long answers for G-NLL baseline evaluation.
"""

import subprocess
import sys
import os
from datetime import datetime

def generate_answers(answer_type, model_name='Llama-3.2-1B', dataset='trivia_qa', 
                     num_samples=400, entity='nikosteam', project='super_guacamole'):
    """Generate answers (short or long) and return the command."""
    os.chdir('src')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'gnll_{answer_type}_{timestamp}'
    
    # Choose brief prompt based on answer type
    if answer_type == 'short':
        brief_prompt = 'short'  # Brief, concise answers
    elif answer_type == 'long':
        brief_prompt = 'detailed'  # Detailed, well-structured answers
    else:
        raise ValueError(f"answer_type must be 'short' or 'long', got {answer_type}")
    
    cmd = [
        'python', 'generate_answers.py',
        '--model_name', model_name,
        '--dataset', dataset,
        '--num_samples', str(num_samples),
        '--num_few_shot', '5',
        '--temperature', '0.0',  # Greedy decoding
        '--num_generations', '1',  # Only one generation per question (greedy)
        '--brief_prompt', brief_prompt,
        '--enable_brief',
        '--brief_always',
        '--no-compute_uncertainties',
        '--no-compute_p_true',
        '--no-get_training_set_generations',
        '--use_context',
        '--entity', entity,
        '--project', project,
        '--experiment_lot', experiment_name
    ]
    
    print(f"\n{'='*80}")
    print(f"Generating {answer_type} answers...")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {answer_type.capitalize()} answers generation completed successfully!")
        print(f"\n⚠️  IMPORTANT: Note the wandb run ID from the output above")
        print(f"   You'll need it for the next steps (evaluation and AUROC computation)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running generation: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted by user")
        return False

def main():
    """Main function."""
    print("="*80)
    print("G-NLL Baseline: Answer Generation")
    print("="*80)
    print()
    print("This script will generate:")
    print("  1. Short answers (brief, concise)")
    print("  2. Long answers (detailed, well-structured)")
    print()
    print("After generation, you'll need to:")
    print("  - Evaluate with LLM judge (using recompute_accuracy_with_judge.py)")
    print("  - Compute AUROC (using compute_gnll_auroc.py)")
    print()
    
    # Configuration
    model_name = 'Llama-3.2-1B'
    dataset = 'trivia_qa'
    num_samples = 400
    entity = 'nikosteam'
    project = 'super_guacamole'
    
    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Num samples: {num_samples}")
    print()
    
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Generate short answers
    print("\n" + "="*80)
    print("STEP 1: Generate Short Answers")
    print("="*80)
    if not generate_answers('short', model_name, dataset, num_samples, entity, project):
        print("Failed to generate short answers. Exiting.")
        sys.exit(1)
    
    input("\nPress Enter to continue to long answers generation...")
    
    # Generate long answers
    print("\n" + "="*80)
    print("STEP 2: Generate Long Answers")
    print("="*80)
    if not generate_answers('long', model_name, dataset, num_samples, entity, project):
        print("Failed to generate long answers. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ Generation Complete!")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Evaluate answers with LLM judge:")
    print("   python recompute_accuracy_with_judge.py <wandb_run_id> llm_llama-3-8b")
    print()
    print("2. Compute G-NLL AUROC:")
    print("   python compute_gnll_auroc.py <path_to_validation_generations.pkl> [--rouge]")
    print()
    print("See GNLL_BASELINE_README.md for detailed instructions.")

if __name__ == '__main__':
    main()

