#!/usr/bin/env python3
"""
Standalone script to compute G-NLL AUROC from existing validation_generations.pkl files.

Usage:
    python compute_gnll_auroc.py <path_to_validation_generations.pkl> [--rouge] [--rouge-threshold 0.3] [--rw-gnll] [--model-name MODEL]
    
Options:
    --rouge: Use ROUGE scores for correctness (for short answers)
    --rouge-threshold: ROUGE-L threshold for correctness (default: 0.3)
    --judge: Use LLM judge accuracy (default, for long answers or when judge was run)
    --rw-gnll: Also compute RW-G-NLL metric (requires --model-name)
    --model-name: Model name used for generation (required for --rw-gnll)
    --similarity-model: Similarity model for RW-G-NLL (default: cross-encoder/stsb-roberta-large)
"""

import argparse
import pickle
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from rouge_score import rouge_scorer

def compute_gnll_auroc(pickle_path, use_rouge=False, rouge_threshold=0.3, use_judge=True,
                       use_rw_gnll=False, similarity_model_name='cross-encoder/stsb-roberta-large',
                       model_name=None, tokenizer=None):
    """
    Compute AUROC for G-NLL baseline and optionally RW-G-NLL.
    
    Args:
        pickle_path: Path to validation_generations.pkl
        use_rouge: If True, use ROUGE scores for correctness (for short answers)
        rouge_threshold: ROUGE-L threshold for correctness (if use_rouge=True)
        use_judge: If True, use LLM judge accuracy (for long answers or when judge was run)
        use_rw_gnll: If True, also compute RW-G-NLL metric
        similarity_model_name: Name of similarity model to use for RW-G-NLL
        model_name: Model name used for generation (needed for tokenizer if use_rw_gnll=True)
        tokenizer: Tokenizer instance (if None and use_rw_gnll=True, will try to load from model_name)
    """
    # Import RW-G-NLL functions if needed
    if use_rw_gnll:
        try:
            # Add src to path for imports
            script_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.join(script_dir, 'src')
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            
            from uncertainty_measures.rw_gnll import (
                initialize_similarity_model,
                compute_rw_gnll
            )
            from transformers import AutoTokenizer
            from models.huggingface_models import get_hf_cache_dir
        except ImportError as e:
            print(f"❌ Error importing RW-G-NLL module: {e}")
            print("Falling back to G-NLL only")
            use_rw_gnll = False
    
    print(f"Loading generations from: {pickle_path}")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    y_true = []
    gnll_uncertainties = []
    rw_gnll_uncertainties = [] if use_rw_gnll else None
    
    # Initialize RW-G-NLL components if needed
    similarity_model = None
    rw_tokenizer = tokenizer
    similarity_cache = {}
    
    if use_rw_gnll:
        print(f"\nInitializing RW-G-NLL components...")
        print(f"  Similarity model: {similarity_model_name}")
        
        # Initialize similarity model
        try:
            similarity_model = initialize_similarity_model(similarity_model_name)
            print(f"  ✅ Similarity model loaded")
        except Exception as e:
            print(f"  ❌ Error loading similarity model: {e}")
            print("  Falling back to G-NLL only")
            use_rw_gnll = False
            similarity_model = None
        
        # Initialize tokenizer if not provided
        if rw_tokenizer is None and model_name:
            print(f"  Loading tokenizer for model: {model_name}")
            try:
                cache_dir = get_hf_cache_dir()
                
                # Determine base path for model
                if 'llama' in model_name.lower():
                    if 'Llama-3' in model_name or 'Llama-3.1' in model_name or 'Meta-Llama-3' in model_name or 'Llama-2' in model_name:
                        base = 'meta-llama'
                    else:
                        base = 'huggyllama'
                    
                    rw_tokenizer = AutoTokenizer.from_pretrained(
                        f"{base}/{model_name}",
                        token_type_ids=None,
                        cache_dir=cache_dir
                    )
                    print(f"  ✅ Tokenizer loaded")
                else:
                    print(f"  ⚠️  Unknown model type, cannot load tokenizer automatically")
                    print(f"  Please provide tokenizer parameter or set use_rw_gnll=False")
                    use_rw_gnll = False
            except Exception as e:
                print(f"  ❌ Error loading tokenizer: {e}")
                print("  Falling back to G-NLL only")
                use_rw_gnll = False
                rw_tokenizer = None
        elif rw_tokenizer is None:
            print(f"  ⚠️  No tokenizer provided and no model_name given")
            print(f"  Cannot compute RW-G-NLL without tokenizer")
            print("  Falling back to G-NLL only")
            use_rw_gnll = False
    
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
            
            # Compute RW-G-NLL if enabled
            if use_rw_gnll and similarity_model is not None and rw_tokenizer is not None:
                try:
                    rw_gnll_score, _ = compute_rw_gnll(
                        entry, similarity_model, rw_tokenizer, cache=similarity_cache
                    )
                    rw_gnll_uncertainties.append(rw_gnll_score)
                except Exception as e:
                    print(f"Warning: Error computing RW-G-NLL for {example_id}: {e}")
                    # Fallback to G-NLL value
                    rw_gnll_uncertainties.append(sequence_nll)
            
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
        # roc_auc_score expects higher scores to predict positive class (correct=1)
        # So we negate G-NLL: higher confidence (lower NLL) should predict correct answers
        # This means we use -gnll_uncertainties as scores
        auroc = roc_auc_score(y_true, -np.array(gnll_uncertainties))
        
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
        
        # Compute RW-G-NLL AUROC if enabled
        if use_rw_gnll and rw_gnll_uncertainties and len(rw_gnll_uncertainties) == len(y_true):
            try:
                rw_gnll_auroc = roc_auc_score(y_true, -np.array(rw_gnll_uncertainties))
                results['RW-G-NLL_AUROC'] = float(rw_gnll_auroc)
                results['Mean_RW-G-NLL'] = float(np.mean(rw_gnll_uncertainties))
                results['Std_RW-G-NLL'] = float(np.std(rw_gnll_uncertainties))
                
                print(f"\nRW-G-NLL Results:")
                print(f"  RW-G-NLL AUROC: {rw_gnll_auroc:.4f}")
                print(f"  Mean RW-G-NLL: {np.mean(rw_gnll_uncertainties):.4f}")
                print(f"  Std RW-G-NLL: {np.std(rw_gnll_uncertainties):.4f}")
            except Exception as e:
                print(f"\n⚠️  Error computing RW-G-NLL AUROC: {e}")
        
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
    parser.add_argument('--rw-gnll', action='store_true',
                       help='Also compute RW-G-NLL metric (requires --model-name)')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name used for generation (required for --rw-gnll)')
    parser.add_argument('--similarity-model', type=str, 
                       default='cross-encoder/stsb-roberta-large',
                       help='Similarity model for RW-G-NLL (default: cross-encoder/stsb-roberta-large)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    use_judge = not args.no_judge and not args.rouge
    
    if args.rw_gnll and not args.model_name:
        print("❌ Error: --rw-gnll requires --model-name")
        sys.exit(1)
    
    results = compute_gnll_auroc(
        args.pickle_path, 
        use_rouge=args.rouge,
        rouge_threshold=args.rouge_threshold,
        use_judge=use_judge,
        use_rw_gnll=args.rw_gnll,
        similarity_model_name=args.similarity_model,
        model_name=args.model_name
    )
    
    if results and args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")

if __name__ == '__main__':
    main()

