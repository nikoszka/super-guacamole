#!/usr/bin/env python3
"""Generate short answers with token tracking."""

import subprocess
import sys
import os
from datetime import datetime


def main():
    os.chdir('src')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'short_answers_token_tracking_{timestamp}'

    cmd = [
        'python', 'generate_answers.py',
        '--model_name', 'Llama-3.1-8B',
        '--dataset', 'trivia_qa',
        '--num_samples', '400',
        '--num_generations', '1',
        '--temperature', '0.0',
        '--model_max_new_tokens', '50',
        '--num_few_shot', '5',
        '--brief_prompt', 'short',
        '--enable_brief',
        # '--metric', 'squad',
        '--use_context',
        '--answerable_only',
        '--get_training_set_generations',
        '--no-compute_p_true',
        '--no-compute_uncertainties',  # ← ADD THIS LINE
        '--entity', 'nikosteam',
        '--project', 'super_guacamole',
        '--experiment_lot', experiment_name
    ]

    print("=" * 80)
    print("🚀 Generating SHORT ANSWERS with Token Tracking")
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