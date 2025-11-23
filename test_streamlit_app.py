"""Quick test to verify the streamlit app can load pickle files correctly.

This script tests the core functionality without launching the full streamlit UI.

Usage:
    python test_streamlit_app.py
"""

import os
import pickle
import sys

# Add src to path
sys.path.insert(0, 'src/analysis')

def test_pickle_loading():
    """Test loading and extracting data from pickle files."""
    
    print("🧪 Testing Streamlit App Components\n")
    print("=" * 60)
    
    # Import the functions from the app
    try:
        from token_visualization_app import load_pickle, extract_pickle_examples
        print("✅ Successfully imported app functions")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Test paths
    test_paths = {
        "Long Answers": "src/boldis/uncertainty/wandb/run-20251121_092732-wiboofpr/files/validation_generations_judge_corrected.pkl",
        "Short Answers": "src/boldis/uncertainty/wandb/run-20251121_011028-sqykmrn7/files/validation_generations.pkl"
    }
    
    for answer_type, pickle_path in test_paths.items():
        print(f"\n📂 Testing {answer_type}")
        print(f"   Path: {pickle_path}")
        
        if not os.path.exists(pickle_path):
            print(f"   ⚠️  File not found (skipping)")
            continue
        
        try:
            # Load pickle
            data = load_pickle(pickle_path)
            print(f"   ✅ Loaded pickle ({len(data)} entries)")
            
            # Extract examples
            examples = extract_pickle_examples(data)
            print(f"   ✅ Extracted {len(examples)} valid examples")
            
            if examples:
                # Check first example
                ex = examples[0]
                print(f"\n   📊 First Example Details:")
                print(f"      - Example ID: {ex['example_id'][:50]}...")
                print(f"      - Question: {ex['question'][:80]}...")
                print(f"      - Correct Answer: {ex['correct_answer'][:50]}...")
                print(f"      - Response: {ex['response'][:80]}...")
                print(f"      - Num Tokens: {len(ex['tokens'])}")
                print(f"      - Accuracy: {ex['accuracy']:.2f}")
                
                # Check data integrity
                assert len(ex['tokens']) == len(ex['nlls']), "Token/NLL length mismatch"
                assert len(ex['tokens']) == len(ex['probs']), "Token/Prob length mismatch"
                assert all(0 <= p <= 1 for p in ex['probs']), "Invalid probabilities"
                assert all(n >= 0 for n in ex['nlls']), "Invalid NLLs"
                
                print(f"      ✅ Data integrity checks passed")
                
                # Show sample token
                if len(ex['tokens']) > 0:
                    print(f"\n   🔍 Sample Token:")
                    idx = 0
                    print(f"      Token: '{ex['tokens'][idx]}'")
                    print(f"      Probability: {ex['probs'][idx]:.4f}")
                    print(f"      NLL: {ex['nlls'][idx]:.3f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\n🚀 Ready to launch the app with:")
    print("   python run_token_viz_app.py")
    
    return True


if __name__ == "__main__":
    success = test_pickle_loading()
    sys.exit(0 if success else 1)


