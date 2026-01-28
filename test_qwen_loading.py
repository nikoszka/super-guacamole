#!/usr/bin/env python3
"""
Quick test script to verify Qwen model loading works correctly.
This tests the model initialization without running a full experiment.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
from models.huggingface_models import HuggingfaceModel

logging.basicConfig(level=logging.INFO)

def test_qwen_model_loading():
    """Test loading Qwen2.5-1.5B model."""
    print("="*80)
    print("Testing Qwen2.5-1.5B model loading...")
    print("="*80)
    
    try:
        # Test basic model loading (FP16)
        print("\n1. Testing FP16 loading (Qwen2.5-1.5B)...")
        model = HuggingfaceModel(
            model_name='Qwen2.5-1.5B',
            stop_sequences=['default'],
            max_new_tokens=50
        )
        print(f"✅ Model loaded successfully!")
        print(f"   Model name: {model.model_name}")
        print(f"   Token limit: {model.token_limit}")
        print(f"   Tokenizer vocab size: {len(model.tokenizer)}")
        
        # Test a simple prediction
        print("\n2. Testing simple prediction...")
        test_prompt = "Question: What is the capital of France?\nAnswer:"
        answer, log_likelihoods, embedding, token_ids, tokens = model.predict(
            test_prompt, temperature=0.0
        )
        print(f"✅ Prediction successful!")
        print(f"   Answer: {answer}")
        print(f"   Number of tokens generated: {len(token_ids)}")
        print(f"   Log-likelihoods: {log_likelihoods[:3]}..." if len(log_likelihoods) > 3 else f"   Log-likelihoods: {log_likelihoods}")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - Qwen model loading works correctly!")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_qwen_model_loading()
    sys.exit(0 if success else 1)
