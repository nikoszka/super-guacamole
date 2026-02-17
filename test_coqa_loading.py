"""Test script to verify CoQA dataset loading."""
import sys
sys.path.append('src')

from data.data_utils import load_ds

def test_coqa_loading():
    """Test CoQA dataset loading and formatting."""
    print("=" * 80)
    print("Testing CoQA Dataset Loading")
    print("=" * 80)
    
    print("\n1. Loading CoQA dataset...")
    try:
        train_dataset, validation_dataset = load_ds('coqa', seed=42)
        print(f"✅ Successfully loaded CoQA dataset")
        print(f"   - Training samples: {len(train_dataset)}")
        print(f"   - Validation samples: {len(validation_dataset)}")
    except Exception as e:
        print(f"❌ Failed to load CoQA dataset: {e}")
        return False
    
    print("\n2. Inspecting sample format...")
    sample = train_dataset[0]
    print(f"   Sample keys: {sample.keys()}")
    print(f"   - ID: {sample['id']}")
    print(f"   - Question: {sample['question'][:100]}...")
    print(f"   - Context length: {len(sample['context'])} chars")
    print(f"   - Answer: {sample['answers']['text'][0][:100]}...")
    
    print("\n3. Checking conversational structure...")
    # Count how many questions come from same story
    story_ids = {}
    for sample in train_dataset[:100]:
        story_id = sample['id'].split('_')[0]
        story_ids[story_id] = story_ids.get(story_id, 0) + 1
    
    avg_questions = sum(story_ids.values()) / len(story_ids)
    print(f"   ✅ Average questions per story: {avg_questions:.1f}")
    print(f"   ✅ Stories in first 100 samples: {len(story_ids)}")
    
    print("\n4. Sample conversation preview:")
    print("-" * 80)
    first_story_id = list(story_ids.keys())[0]
    story_samples = [s for s in train_dataset[:100] if s['id'].startswith(first_story_id)]
    
    print(f"Story: {story_samples[0]['context'][:200]}...\n")
    for i, sample in enumerate(story_samples[:3], 1):
        print(f"Q{i}: {sample['question']}")
        print(f"A{i}: {sample['answers']['text'][0]}\n")
    print("-" * 80)
    
    print("\n5. Validation dataset check...")
    val_sample = validation_dataset[0]
    print(f"   ✅ Validation sample ID: {val_sample['id']}")
    print(f"   ✅ Validation question: {val_sample['question'][:80]}...")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - CoQA dataset is ready to use!")
    print("=" * 80)
    print("\nYou can now run: ./run_CoQA_all.sh")
    return True

if __name__ == '__main__':
    success = test_coqa_loading()
    sys.exit(0 if success else 1)
