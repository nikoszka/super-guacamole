import pickle
import sys
import torch

path = r"C:\Users\nikos\PycharmProjects\nllSAR\src\boldis\uncertainty\wandb\run-20251121_190025-5qvhbs97\files\uncertainty_measures.pkl"
try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print("Type of data:", type(data))
    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
        # Check if 'accuracies' or something similar exists
        for k in data.keys():
            print(f"Key: {k}, Type: {type(data[k])}")
    else:
        print("Data is not a dict")
except Exception as e:
    print(e)

