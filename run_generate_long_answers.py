#!/usr/bin/env python3
"""Generate long answers with token tracking."""

import subprocess
import sys
import os
from datetime import datetime


def main():
    os.chdir('src')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'long_answers_token_tracking_{timestamp}'

    cmd = [
        'python', 'generate_answers.py',
        '--model_name', 'Llama-3.2-1B',
        '--dataset', 'trivia_qa',
        '--num_samples', '400',
        '--num_generations', '1',
        '--temperature', '0.0',
        '--model_max_new_tokens', '100',
        '--num_few_shot', '0',
        '--brief_prompt', 'manual',
        '--enable_brief',
        # '--metric', 'llm_llama-3-8b',
        '--no-use_context',  # ← Enable context for long answers
        '--answerable_only',
        '--get_training_set_generations',
        '--no-compute_p_true',
        '--no-compute_uncertainties',  # ← ADD THIS LINE
        '--entity', 'nikosteam',
        '--project', 'super_guacamole',
        '--experiment_lot', experiment_name
    ]

    print("=" * 80)
    print("🚀 Generating LONG ANSWERS with Token Tracking")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print("=" * 80)
    print()

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Generation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()