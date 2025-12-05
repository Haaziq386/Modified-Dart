import torch
import os

# Path to where you extracted the EMG.zip
# e.g., datasets/EMG/
data_path = 'datasets/FDB/' 

files = ['train.pt', 'val.pt', 'test.pt']

for f in files:
    path = os.path.join(data_path, f)
    if os.path.exists(path):
        data = torch.load(path)
        # The paper says keys are 'samples' and 'labels'
        # Shape: [num_samples, num_channels, seq_len]
        samples = data['samples']
        labels = data['labels']
        
        print(f"=== {f} ===")
        print(f"Data loaded from: {data_path}")
        print(f"Samples shape: {samples.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {torch.unique(labels)}")
    else:
        print(f"File not found: {path}")