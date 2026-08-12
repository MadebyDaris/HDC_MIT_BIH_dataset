import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
import torchhd
import os
import numpy as np

# AAMI classes mapping
# MIT-BIH symbols to AAMI classes
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular ectopic
    'V': 2, 'E': 2,                          # Ventricular ectopic
    'F': 3,                                  # Fusion
    '/': 4, 'f': 4, 'Q': 4                   # Unknown
}

class MITBIHDataset(Dataset):
    def __init__(self, data_dir="./physionet.org/files/mitdb/1.0.0", window_size=360):
        self.data_dir = data_dir
        self.window_size = window_size
        self.half_window = window_size // 2
        self.samples = []
        self.labels = []
        self.preprocess()

    def preprocess(self):
        records_path = os.path.join(self.data_dir, "RECORDS")
        with open(records_path, 'r') as f:
            records = [line.strip() for line in f if line.strip()]

        print(f"Found {len(records)} records. Processing...")
        
        for patient in records:
            record_path = os.path.join(self.data_dir, patient)
            # Load the record and annotations
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            # We typically use the first channel (usually MLII)
            signal = record.p_signal[:, 0]
            
            # Get the locations (sample indices) and symbols of the heartbeats
            peaks = annotation.sample
            symbols = annotation.symbol
            
            for peak, symbol in zip(peaks, symbols):
                # Check if the symbol is in our mapping
                if symbol in AAMI_MAPPING:
                    # Ensure we can extract a full window around the peak
                    if peak >= self.half_window and peak + self.half_window < len(signal):
                        # Extract window
                        window = signal[peak - self.half_window : peak + self.half_window]
                        
                        # Normalize the window (Z-score normalization)
                        mean = np.mean(window)
                        std = np.std(window)
                        if std > 0:
                            window = (window - mean) / std
                        else:
                            window = window - mean
                            
                        self.samples.append(window)
                        self.labels.append(AAMI_MAPPING[symbol])
        
        # Convert to PyTorch tensors
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        print(f"Preprocessing completed. Extracted {len(self.samples)} heartbeats.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

if __name__ == "__main__":
    print("Loading MIT-BIH dataset...")
    dataset = MITBIHDataset()
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check class distribution
    labels = dataset.labels.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
    
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"{class_names[u]} (Class {u}): {c}")
        
    # Create a DataLoader for standard usage
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get a batch
    batch_signals, batch_labels = next(iter(dataloader))
    print(f"\nBatch signals shape: {batch_signals.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")


def load_and_inspect_data():
    # Load the dataset
    dataset = MITBIHDataset()
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check class distribution
    labels = dataset.labels.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
    
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"{class_names[u]} (Class {u}): {c}")

    # Get a batch
    batch_signals, batch_labels = next(iter(dataloader))
    print(f"\nBatch signals shape: {batch_signals.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")

    encoder_model = hdc_db.encoder(batch_signals[0], batch_labels[0], 360)
    
    for i in range(len(batch_signals)):
        batch_signals[i] = encoder_model(batch_signals[i], batch_labels[i], 360)
    
    print(batch_signals[0])