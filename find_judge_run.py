import pickle
import os
from pathlib import Path
import sys

def check_pkl(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            if 'validation_is_false' in data:
                return True, len(data['validation_is_false'])
            if 'p_false' in data:
                return False, "Has p_false but no validation_is_false"
        return False, "No relevant keys"
    except Exception as e:
        return False, str(e)

base_dir = Path(r"C:\Users\nikos\PycharmProjects\nllSAR\src\boldis\uncertainty\wandb")
print(f"Searching in {base_dir}...")

found_runs = []

for root, dirs, files in os.walk(base_dir):
    if 'uncertainty_measures.pkl' in files:
        path = Path(root) / 'uncertainty_measures.pkl'
        has_judge, info = check_pkl(path)
        if has_judge:
            run_id = path.parent.parent.name
            print(f"FOUND JUDGE DATA in {run_id}: {info} examples")
            found_runs.append((run_id, path))

if not found_runs:
    print("No runs found with 'validation_is_false' in uncertainty_measures.pkl")
else:
    print("\nRecommended usage:")
    latest_run = sorted(found_runs, key=lambda x: x[0], reverse=True)[0]
    print(f"Use: {latest_run[1]}")

