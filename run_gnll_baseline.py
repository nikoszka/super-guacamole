#!/usr/bin/env python3
"""
Script to generate short and long answers, evaluate them, and compute AUROC for G-NLL baseline.
- Short answers: evaluated with ROUGE and LLM-as-a-judge
- Long answers: evaluated with LLM-as-a-judge only
- G-NLL: sum of token log-probs from greedy answer
"""

import subprocess
import sys
import os
from datetime import datetime
import pickle
import json

def run_generation(model_name, dataset, answer_type, num_samples=400, entity='nikosteam', project='super_guacamole'):
    """Generate answers (short or long) and return wandb run ID."""
    os.chdir('src')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'gnll_baseline_{answer_type}_{timestamp}'
    
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Generation completed successfully!")
        
        # Try to extract wandb run ID from output
        # The run ID is typically logged or can be found in wandb
        # For now, we'll need to get it from wandb or the latest run
        print("Note: Please note the wandb run ID from the output above")
        return None  # Will need to be set manually or extracted from wandb
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running generation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)

def evaluate_with_judge(wandb_runid, metric, entity='nikosteam', project='super_guacamole'):
    """Evaluate answers using LLM-as-a-judge."""
    os.chdir('src')
    
    cmd = [
        'python', 'compute_uncertainty_measures.py',
        '--eval_wandb_runid', wandb_runid,
        '--metric', metric,
        '--recompute_accuracy',
        '--no-compute_predictive_entropy',
        '--no-compute_p_ik',
        '--no-compute_context_entails_response',
        '--no-analyze_run',
        '--entity', entity,
        '--project', project,
        '--restore_entity_eval', entity,
    ]
    
    print(f"\n{'='*80}")
    print(f"Evaluating with {metric} judge...")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error evaluating: {e}")
        sys.exit(1)

def compute_gnll_auroc(pickle_path, use_rouge=False, rouge_threshold=0.3, 
                       use_rw_gnll=False, similarity_model_name='cross-encoder/stsb-roberta-large',
                       model_name=None, tokenizer=None):
    """
    Compute AUROC for G-NLL baseline and optionally RW-G-NLL.
    
    Args:
        pickle_path: Path to validation_generations.pkl
        use_rouge: If True, use ROUGE scores for correctness (for short answers)
                   If False, use LLM judge accuracy (for long answers)
        rouge_threshold: ROUGE-L threshold for correctness (if use_rouge=True)
        use_rw_gnll: If True, also compute RW-G-NLL metric
        similarity_model_name: Name of similarity model to use for RW-G-NLL
        model_name: Model name used for generation (needed for tokenizer if use_rw_gnll=True)
        tokenizer: Tokenizer instance (if None and use_rw_gnll=True, will try to load from model_name)
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from rouge_score import rouge_scorer
    
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
    
    print(f"\n{'='*80}")
    print(f"Computing G-NLL AUROC...")
    print(f"{'='*80}")
    print(f"Loading generations from: {pickle_path}")
    
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
        print(f"Using ROUGE-L threshold: {rouge_threshold}")
    else:
        print("Using LLM judge accuracy (from 'accuracy' field)")
    
    for example_id, entry in data.items():
        if 'most_likely_answer' not in entry:
            continue
            
        mla = entry['most_likely_answer']
        pred_answer = mla.get('response', '').strip()
        
        if use_rouge:
            # Use ROUGE score for correctness
            if 'reference' in entry and 'answers' in entry['reference']:
                true_answers = entry['reference']['answers']['text']
                best_rougeL = 0.0
                for ref in true_answers:
                    score = scorer.score(ref.strip(), pred_answer)
                    best_rougeL = max(best_rougeL, score['rougeL'].fmeasure)
                is_correct = int(best_rougeL >= rouge_threshold)
            else:
                continue
        else:
            # Use LLM judge accuracy
            is_correct = int(mla.get('accuracy', 0.0) > 0.5)
        
        # Compute G-NLL (sum of token log-probs, negated)
        if 'token_log_likelihoods' in mla:
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
            print(f"Warning: No token_log_likelihoods for example {example_id}")
    
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
            'G-NLL_AUROC': auroc,
            'Accuracy': accuracy,
            'Num_examples': len(y_true),
            'Mean_G-NLL': np.mean(gnll_uncertainties),
            'Std_G-NLL': np.std(gnll_uncertainties),
        }
        
        print(f"\nResults:")
        print(f"  G-NLL AUROC: {auroc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Number of examples: {len(y_true)}")
        print(f"  Mean G-NLL: {np.mean(gnll_uncertainties):.4f}")
        print(f"  Std G-NLL: {np.std(gnll_uncertainties):.4f}")
        
        # Compute RW-G-NLL AUROC if enabled
        if use_rw_gnll and rw_gnll_uncertainties and len(rw_gnll_uncertainties) == len(y_true):
            try:
                rw_gnll_auroc = roc_auc_score(y_true, -np.array(rw_gnll_uncertainties))
                results['RW-G-NLL_AUROC'] = rw_gnll_auroc
                results['Mean_RW-G-NLL'] = np.mean(rw_gnll_uncertainties)
                results['Std_RW-G-NLL'] = np.std(rw_gnll_uncertainties)
                
                print(f"\nRW-G-NLL Results:")
                print(f"  RW-G-NLL AUROC: {rw_gnll_auroc:.4f}")
                print(f"  Mean RW-G-NLL: {np.mean(rw_gnll_uncertainties):.4f}")
                print(f"  Std RW-G-NLL: {np.std(rw_gnll_uncertainties):.4f}")
            except Exception as e:
                print(f"\n⚠️  Error computing RW-G-NLL AUROC: {e}")
        
        return results
        
    except ValueError as e:
        print(f"❌ Error computing AUROC: {e}")
        print(f"This might happen if all labels are the same (all correct or all incorrect)")
        return None

def main():
    """Main function to run the full pipeline."""
    print("="*80)
    print("G-NLL Baseline Evaluation Pipeline")
    print("="*80)
    print()
    print("This script will:")
    print("  1. Generate short answers (evaluated with ROUGE and LLM judge)")
    print("  2. Generate long answers (evaluated with LLM judge only)")
    print("  3. Compute AUROC for G-NLL baseline for both")
    print()
    
    # Configuration
    model_name = 'Llama-3.2-1B'
    dataset = 'trivia_qa'
    num_samples = 400
    judge_metric = 'llm_llama-3-8b'  # LLM judge to use
    entity = 'nikosteam'
    project = 'super_guacamole'
    
    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Num samples: {num_samples}")
    print(f"  Judge metric: {judge_metric}")
    print()
    
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    results = {}
    
    # Step 1: Generate short answers
    print("\n" + "="*80)
    print("STEP 1: Generate Short Answers")
    print("="*80)
    short_runid = run_generation(model_name, dataset, 'short', num_samples, entity, project)
    print("\n⚠️  Please note the wandb run ID from the output above")
    short_runid = input("Enter the wandb run ID for short answers generation: ").strip()
    
    # Step 2: Evaluate short answers with LLM judge
    print("\n" + "="*80)
    print("STEP 2: Evaluate Short Answers with LLM Judge")
    print("="*80)
    evaluate_with_judge(short_runid, judge_metric, entity, project)
    
    # Step 3: Generate long answers
    print("\n" + "="*80)
    print("STEP 3: Generate Long Answers")
    print("="*80)
    long_runid = run_generation(model_name, dataset, 'long', num_samples, entity, project)
    print("\n⚠️  Please note the wandb run ID from the output above")
    long_runid = input("Enter the wandb run ID for long answers generation: ").strip()
    
    # Step 4: Evaluate long answers with LLM judge
    print("\n" + "="*80)
    print("STEP 4: Evaluate Long Answers with LLM Judge")
    print("="*80)
    evaluate_with_judge(long_runid, judge_metric, entity, project)
    
    # Step 5: Compute AUROC for short answers (ROUGE and LLM judge)
    print("\n" + "="*80)
    print("STEP 5: Compute G-NLL AUROC for Short Answers")
    print("="*80)
    
    # Find the pickle file - it should be in the wandb run directory
    # For now, we'll ask the user for the path
    print("\nTo compute AUROC, we need the validation_generations.pkl file.")
    print("This is typically located in: src/nikos/uncertainty/wandb/run-<ID>/files/validation_generations.pkl")
    
    short_pickle = input(f"Enter path to short answers validation_generations.pkl (or press Enter to skip): ").strip()
    if short_pickle and os.path.exists(short_pickle):
        print("\n--- Short Answers: ROUGE-based correctness ---")
        results['short_rouge'] = compute_gnll_auroc(
            short_pickle, use_rouge=True, rouge_threshold=0.3,
            use_rw_gnll=False, model_name=model_name
        )
        
        # Also compute with LLM judge
        print("\n--- Short Answers: LLM Judge-based correctness ---")
        results['short_judge'] = compute_gnll_auroc(
            short_pickle, use_rouge=False,
            use_rw_gnll=False, model_name=model_name
        )
    else:
        print("Skipping short answers AUROC computation")
    
    # Step 6: Compute AUROC for long answers (LLM judge only)
    print("\n" + "="*80)
    print("STEP 6: Compute G-NLL AUROC for Long Answers")
    print("="*80)
    
    long_pickle = input(f"Enter path to long answers validation_generations.pkl (or press Enter to skip): ").strip()
    if long_pickle and os.path.exists(long_pickle):
        # Ask if user wants to compute RW-G-NLL for long answers
        use_rw_gnll_long = input("Compute RW-G-NLL for long answers? (y/n, default=n): ").strip().lower() == 'y'
        
        print("\n--- Long Answers: LLM Judge-based correctness ---")
        results['long_judge'] = compute_gnll_auroc(
            long_pickle, use_rouge=False,
            use_rw_gnll=use_rw_gnll_long, model_name=model_name
        )
    else:
        print("Skipping long answers AUROC computation")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    if 'short_rouge' in results and results['short_rouge']:
        print(f"\nShort Answers (ROUGE-based):")
        print(f"  G-NLL AUROC: {results['short_rouge']['G-NLL_AUROC']:.4f}")
        print(f"  Accuracy: {results['short_rouge']['Accuracy']:.4f}")
    
    if 'short_judge' in results and results['short_judge']:
        print(f"\nShort Answers (LLM Judge-based):")
        print(f"  G-NLL AUROC: {results['short_judge']['G-NLL_AUROC']:.4f}")
        print(f"  Accuracy: {results['short_judge']['Accuracy']:.4f}")
    
    if 'long_judge' in results and results['long_judge']:
        print(f"\nLong Answers (LLM Judge-based):")
        print(f"  G-NLL AUROC: {results['long_judge']['G-NLL_AUROC']:.4f}")
        print(f"  Accuracy: {results['long_judge']['Accuracy']:.4f}")
        if 'RW-G-NLL_AUROC' in results['long_judge']:
            print(f"  RW-G-NLL AUROC: {results['long_judge']['RW-G-NLL_AUROC']:.4f}")
    
    # Save results to JSON
    output_file = 'gnll_baseline_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Results saved to {output_file}")

if __name__ == '__main__':
    main()

