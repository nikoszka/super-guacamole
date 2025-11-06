#!/usr/bin/env python3
"""
Standalone script to compute G-NLL AUROC from existing validation_generations.pkl files.

Usage:
    python compute_gnll_auroc.py <path_to_validation_generations.pkl> [--rouge] [--rouge-threshold 0.3]
    
Options:
    --rouge: Use ROUGE scores for correctness (for short answers)
    --rouge-threshold: ROUGE-L threshold for correctness (default: 0.3)
    --judge: Use LLM judge accuracy (default, for long answers or when judge was run)
"""

import argparse
import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from rouge_score import rouge_scorer

def compute_gnll_auroc(pickle_path, use_rouge=False, rouge_threshold=0.3, use_judge=True):
    """
    Compute AUROC for G-NLL baseline.
    
    Args:
        pickle_path: Path to validation_generations.pkl
        use_rouge: If True, use ROUGE scores for correctness (for short answers)
        rouge_threshold: ROUGE-L threshold for correctness (if use_rouge=True)
        use_judge: If True, use LLM judge accuracy (for long answers or when judge was run)
    """
    print(f"Loading generations from: {pickle_path}")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    y_true = []
    gnll_uncertainties = []
    
    if use_rouge:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        print(f"Using ROUGE-L threshold: {rouge_threshold} for correctness")
    elif use_judge:
        print("Using LLM judge accuracy (from 'accuracy' field in most_likely_answer)")
    else:
        print("Using exact match with reference answers")
    
    skipped = 0
    for example_id, entry in data.items():
        if 'most_likely_answer' not in entry:
            skipped += 1
            continue
            
        mla = entry['most_likely_answer']
        pred_answer = mla.get('response', '').strip()
        
        if use_rouge:
            # Use ROUGE score for correctness
            if 'reference' in entry and 'answers' in entry['reference']:
                true_answers = entry['reference']['answers']['text']
                if not true_answers:
                    skipped += 1
                    continue
                best_rougeL = 0.0
                for ref in true_answers:
                    score = scorer.score(ref.strip(), pred_answer)
                    best_rougeL = max(best_rougeL, score['rougeL'].fmeasure)
                is_correct = int(best_rougeL >= rouge_threshold)
            else:
                skipped += 1
                continue
        elif use_judge:
            # Use LLM judge accuracy
            accuracy = mla.get('accuracy', None)
            if accuracy is None:
                skipped += 1
                continue
            is_correct = int(accuracy > 0.5)
        else:
            # Use exact match
            if 'reference' in entry and 'answers' in entry['reference']:
                true_answers = [ans.strip().lower() for ans in entry['reference']['answers']['text']]
                is_correct = int(pred_answer.lower() in true_answers)
            else:
                skipped += 1
                continue
        
        # Compute G-NLL (sum of token log-probs, negated)
        if 'token_log_likelihoods' in mla and len(mla['token_log_likelihoods']) > 0:
            token_log_likelihoods = mla['token_log_likelihoods']
            sequence_nll = -sum(token_log_likelihoods)  # Negative log likelihood
            gnll_uncertainties.append(sequence_nll)
            y_true.append(is_correct)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} examples (missing data)")
    
    if len(y_true) == 0:
        print("❌ No valid examples found!")
        return None
    
    # Compute AUROC
    try:
        # For AUROC, higher uncertainty (higher NLL) should predict incorrect answers
        # So we use gnll_uncertainties directly (higher = more uncertain = more likely wrong)
        auroc = roc_auc_score(y_true, gnll_uncertainties)
        
        accuracy = sum(y_true) / len(y_true) if len(y_true) > 0 else 0
        
        results = {
            'G-NLL_AUROC': float(auroc),
            'Accuracy': float(accuracy),
            'Num_examples': len(y_true),
            'Num_correct': int(sum(y_true)),
            'Num_incorrect': int(len(y_true) - sum(y_true)),
            'Mean_G-NLL': float(np.mean(gnll_uncertainties)),
            'Std_G-NLL': float(np.std(gnll_uncertainties)),
            'Min_G-NLL': float(np.min(gnll_uncertainties)),
            'Max_G-NLL': float(np.max(gnll_uncertainties)),
        }
        
        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"  G-NLL AUROC: {auroc:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({sum(y_true)}/{len(y_true)})")
        print(f"  Number of examples: {len(y_true)}")
        print(f"  Mean G-NLL: {np.mean(gnll_uncertainties):.4f}")
        print(f"  Std G-NLL: {np.std(gnll_uncertainties):.4f}")
        print(f"  Min G-NLL: {np.min(gnll_uncertainties):.4f}")
        print(f"  Max G-NLL: {np.max(gnll_uncertainties):.4f}")
        print(f"{'='*80}")
        
        return results
        
    except ValueError as e:
        print(f"❌ Error computing AUROC: {e}")
        print(f"This might happen if all labels are the same (all correct or all incorrect)")
        print(f"Correct: {sum(y_true)}, Incorrect: {len(y_true) - sum(y_true)}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Compute G-NLL AUROC from validation_generations.pkl'
    )
    parser.add_argument('pickle_path', type=str, help='Path to validation_generations.pkl')
    parser.add_argument('--rouge', action='store_true', 
                       help='Use ROUGE scores for correctness (for short answers)')
    parser.add_argument('--rouge-threshold', type=float, default=0.3,
                       help='ROUGE-L threshold for correctness (default: 0.3)')
    parser.add_argument('--no-judge', action='store_true',
                       help='Do not use LLM judge accuracy (use exact match instead)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    use_judge = not args.no_judge and not args.rouge
    
    results = compute_gnll_auroc(
        args.pickle_path, 
        use_rouge=args.rouge,
        rouge_threshold=args.rouge_threshold,
        use_judge=use_judge
    )
    
    if results and args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")

if __name__ == '__main__':
    main()

