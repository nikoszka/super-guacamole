#!/usr/bin/env python3
"""
Test script to validate SQuAD dataset loading and integration.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.data_utils import load_ds

def test_squad_loading():
    """Test loading SQuAD dataset."""
    print("="*80)
    print("Testing SQuAD Dataset Loading")
    print("="*80)
    
    try:
        print("\n1. Loading SQuAD v2 dataset...")
        train_dataset, validation_dataset = load_ds('squad', seed=10)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Train size: {len(train_dataset)}")
        print(f"   Validation size: {len(validation_dataset)}")
        
        # Check structure of a sample
        print("\n2. Validating dataset structure...")
        sample = validation_dataset[0]
        required_keys = ['question', 'context', 'answers', 'id']
        
        print(f"   Sample keys: {list(sample.keys())}")
        
        for key in required_keys:
            if key in sample:
                print(f"   ✅ '{key}' field present")
            else:
                print(f"   ❌ '{key}' field MISSING")
                return False
        
        # Show sample data
        print("\n3. Sample data from SQuAD:")
        print(f"   ID: {sample['id']}")
        print(f"   Question: {sample['question'][:100]}...")
        print(f"   Context: {sample['context'][:150]}...")
        print(f"   Answers: {sample['answers']['text'][:3]}")
        
        # Check answerable vs unanswerable questions
        print("\n4. Checking answerable/unanswerable split...")
        answerable = sum(1 for s in validation_dataset if len(s['answers']['text']) > 0)
        unanswerable = len(validation_dataset) - answerable
        print(f"   Answerable: {answerable}")
        print(f"   Unanswerable: {unanswerable}")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - SQuAD dataset integration verified!")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_squad_loading()
    sys.exit(0 if success else 1)
