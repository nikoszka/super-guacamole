#!/usr/bin/env python3
"""
Update validation_generations.pkl with LLM judge accuracy scores
from the uncertainty_measures.pkl file created by recompute_accuracy.
"""

import pickle
import argparse
from pathlib import Path
import numpy as np
import torch  # Needed for unpickling

def update_pickle_with_uncertainty_measures(
    original_pickle_path: str,
    uncertainty_measures_path: str,
    output_pickle_path: str = None
):
    """
    Update a validation_generations.pkl file with judge accuracy from uncertainty_measures.pkl.
    
    Args:
        original_pickle_path: Path to original validation_generations.pkl
        uncertainty_measures_path: Path to uncertainty_measures.pkl from recompute_accuracy run
        output_pickle_path: Path to save updated pickle (optional)
    """
    # Load original pickle
    print(f"Loading original pickle: {original_pickle_path}")
    with open(original_pickle_path, 'rb') as f:
        validation_data = pickle.load(f)
    
    # Load uncertainty measures (contains judge results)
    print(f"Loading uncertainty measures: {uncertainty_measures_path}")
    with open(uncertainty_measures_path, 'rb') as f:
        uncertainty_data = pickle.load(f)
    
    # Extract validation_is_false (0.0 = correct, 1.0 = incorrect)
    try:
        validation_is_false = uncertainty_data['validation_is_false']
    except KeyError:
        print(f"Error: 'validation_is_false' key not found in uncertainty_measures.pkl.")
        print(f"Available keys: {list(uncertainty_data.keys())}")
        print("Note: If the keys list is empty, you might be pointing to an empty pickle file generated")
        print("during the initial generation run. You need the one from the 'recompute_accuracy' run.")
        raise
    
    # Convert to accuracy (1.0 = correct, 0.0 = incorrect)
    judge_accuracies = [1.0 - is_false for is_false in validation_is_false]
    
    print(f"Found {len(judge_accuracies)} judge labels")
    print(f"Validation data has {len(validation_data)} examples")
    
    if len(judge_accuracies) != len(validation_data):
        raise ValueError(
            f"Mismatch: {len(judge_accuracies)} judge labels vs "
            f"{len(validation_data)} validation examples"
        )
    
    # Count correct/incorrect before and after
    before_correct = 0
    after_correct = 0
    
    # Update accuracy in validation_data
    # Both should be in the same order since they came from the same generation run
    updated_count = 0
    for i, (example_id, example) in enumerate(validation_data.items()):
        if 'most_likely_answer' in example:
            old_accuracy = example['most_likely_answer'].get('accuracy', None)
            new_accuracy = judge_accuracies[i]
            
            if old_accuracy is not None and old_accuracy > 0.5:
                before_correct += 1
            if new_accuracy > 0.5:
                after_correct += 1
            
            example['most_likely_answer']['accuracy'] = new_accuracy
            updated_count += 1
            
            if updated_count <= 5:  # Show first 5 examples
                print(f"  Example {i} ({example_id[:40]}...): "
                      f"{old_accuracy:.3f} -> {new_accuracy:.3f}")
    
    print(f"\nUpdated {updated_count} examples with judge scores")
    print(f"Correct answers BEFORE judge: {before_correct}/{updated_count} "
          f"({100*before_correct/updated_count:.1f}%)")
    print(f"Correct answers AFTER judge: {after_correct}/{updated_count} "
          f"({100*after_correct/updated_count:.1f}%)")
    
    # Determine output path
    if output_pickle_path is None:
        original_path = Path(original_pickle_path)
        output_pickle_path = original_path.parent / f"{original_path.stem}_judge_corrected.pkl"
    
    # Save updated pickle
    print(f"\nSaving updated pickle to: {output_pickle_path}")
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(validation_data, f)
    
    print("Done! Judge-corrected pickle saved successfully.")
    return str(output_pickle_path)

def main():
    parser = argparse.ArgumentParser(
        description='Update validation_generations.pkl with judge scores from uncertainty_measures.pkl'
    )
    parser.add_argument('--original-pickle', type=str, required=True,
                       help='Path to original validation_generations.pkl')
    parser.add_argument('--uncertainty-measures', type=str, required=True,
                       help='Path to uncertainty_measures.pkl from judge run')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (optional)')
    
    args = parser.parse_args()
    
    update_pickle_with_uncertainty_measures(
        args.original_pickle,
        args.uncertainty_measures,
        args.output
    )

if __name__ == '__main__':
    main()

