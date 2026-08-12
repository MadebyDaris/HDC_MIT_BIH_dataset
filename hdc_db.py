import torch
import torchhd
from torchhd import embeddings
from create_db import MITBIHDataset
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

class HDCClassifier(torch.nn.Module):
    def __init__(self, dimensions: int, num_classes: int = 5):
        super(HDCClassifier, self).__init__()
        self.dimensions = dimensions
        self.num_classes = num_classes

        # Class hypervectors
        self.class_hvs = torch.nn.Parameter(torch.randn(num_classes, dimensions, dtype=torch.float32))

        # Initialize class hypervectors to be orthogonal
        torch.nn.init.orthogonal_(self.class_hvs)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Calculate similarity (logits) between encoded signals and class hypervectors
        similarities = torchhd.cosine_similarity(data, self.class_hvs)
        
        # Note: We return similarities (logits) for the CrossEntropyLoss, not the argmax!
        return similarities

    def train_model(self, dataloader: torch.utils.data.DataLoader, epochs: int = 10, learning_rate: float = 0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for i, (encoded_signals, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                logits = self(encoded_signals)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    def test_model(self, dataloader: torch.utils.data.DataLoader):
        correct = 0
        total = 0
        with torch.no_grad():
            for encoded_signals, labels in dataloader:
                logits = self(encoded_signals)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        return correct / total


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

def test_hdc_database(pt_file="hdc_dataset.pt"):
    print(f"Loading {pt_file}...")
    data = torch.load(pt_file, weights_only=False)
    
    samples = data['samples']
    labels = data['labels']
    dimensions = data['dimensions']
    
    print(f"Loaded {len(samples)} samples with {dimensions} dimensions.")
    
    # Simple train/test split (80/20)
    dataset = torch.utils.data.TensorDataset(samples, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # We can use large batch sizes because we are using pre-encoded hypervectors!
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print("Initializing HDC Classifier...")
    model = HDCClassifier(dimensions=dimensions, num_classes=5)
    
    print("Training model on encoded hypervectors...")
    model.train_model(train_loader, epochs=5)
    
    print("Testing model...")
    accuracy = model.test_model(test_loader)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # create_hdc_database(dimensions=1000, num_levels=100, output_file="hdc_dataset.pt")
    test_hdc_database(pt_file="hdc_dataset.pt")
