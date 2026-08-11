import torch
import torchhd
from torchhd import embeddings
from main import MITBIHDataset
import os

class HDCEncoder(torch.nn.Module):
    def __init__(self, dimensions: int = 10000, num_levels: int = 100, window_size: int = 360, vsa: str = "MAP"):
        super(HDCEncoder, self).__init__()
        self.dimensions = dimensions
        self.num_levels = num_levels
        self.window_size = window_size
        
        self.level = embeddings.Level(num_levels, dimensions, vsa=vsa)
        
        # Position embedding for the time step index
        self.position = embeddings.Random(window_size, dimensions, vsa=vsa)


    def forward(self, signal: torch.Tensor) -> torch.Tensor:        
        clamped_signal = torch.clamp(signal, min=-5.0, max=5.0)
        
        # Normalize to [0, 1] then scale to [0, num_levels - 1]
        normalized_signal = (clamped_signal + 5.0) / 10.0
        level_indices = (normalized_signal * (self.num_levels - 1)).long()
        
        level_hvs = self.level(level_indices)
        
        pos_indices = torch.arange(self.window_size, device=signal.device)
        
        pos_hvs = self.position(pos_indices)
        
        bound_hvs = torchhd.bind(level_hvs, pos_hvs.unsqueeze(0))
        
        encoded_hv = torchhd.multiset(bound_hvs)
        
        return encoded_hv


def create_hdc_database(dimensions=10000, num_levels=100, output_file="hdc_dataset.pt"):
    print("Loading base MIT-BIH dataset...")
    base_dataset = MITBIHDataset()
    
    dataloader = torch.utils.data.DataLoader(base_dataset, batch_size=1024, shuffle=False)
    
    encoder = HDCEncoder(dimensions=dimensions, num_levels=num_levels, window_size=base_dataset.window_size)
    
    print(f"Encoding dataset into HDC (d={dimensions}, levels={num_levels})...")
    encoded_samples = []
    all_labels = []
    
    # Use torch.no_grad() as we don't need gradients for encoding
    with torch.no_grad():
        for i, (signals, labels) in enumerate(dataloader):
            encoded_batch = encoder(signals)
            encoded_samples.append(encoded_batch)
            all_labels.append(labels)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} batches...")
            
    final_samples = torch.cat(encoded_samples, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    
    print(f"Finished encoding. Final tensor shape: {final_samples.shape}")
    
    print(f"Saving to {output_file}...")
    torch.save({
        'samples': final_samples,
        'labels': final_labels,
        'dimensions': dimensions,
        'num_levels': num_levels
    }, output_file)
    print("Saved successfully!")

if __name__ == "__main__":
    create_hdc_database(dimensions=10000, num_levels=100, output_file="hdc_dataset.pt")
