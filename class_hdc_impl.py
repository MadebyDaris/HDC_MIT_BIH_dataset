"""
class_hdc_impl.py

Standard HDC classification using torchhd.models.Centroid.

Workflow:
  1. Load the pre-encoded hdc_dataset.pt hypervectors.
  2. Train a torchhd.models.Centroid model (bundles training samples into
     class prototype hypervectors – true HDC, no gradient descent).
  3. Export the class prototype weight vectors to disk.
  4. Reload the prototypes and benchmark five distance metrics:
       - Cosine similarity  (default HDC)
       - Dot-product similarity
       - Hamming distance   (bit-flip count, ideal for binary/bipolar)
       - Euclidean distance
       - Manhattan distance
  5. Print per-class and overall metrics (accuracy, precision, recall, F1).
"""

import torch
import torchhd
from torchhd import models
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]


# Load pre-encoded dataset
def load_hdc_dataset(pt_file: str = "hdc_dataset.pt"):
    print(f"Loading {pt_file}...")
    data = torch.load(pt_file, weights_only=False)
    samples   = data["samples"]     # (N, D)  float / MAPTensor
    labels    = data["labels"]      # (N,)    long
    dimensions = data["dimensions"]
    print(f"  {len(samples)} samples, {dimensions} dimensions, {labels.max().item()+1} classes")
    return samples, labels, dimensions


# Train Centroid model – the canonical HDC way
def train_centroid(train_samples: torch.Tensor, train_labels: torch.Tensor,
                   dimensions: int, num_classes: int = 5) -> models.Centroid:
    """
    torchhd.models.Centroid accumulates one prototype per class by bundling.
    This is a single-pass, no-backprop operation exactly how HDC is meant to learn.
    """
    print("Training Centroid model...")
    model = models.Centroid(dimensions, num_classes)

    # Feed the training set in one batch (Centroid is O(N*D), very fast)
    model.train()
    model.add(train_samples, train_labels)

    # Normalise the prototypes so cosine similarity is well-defined
    model.normalize()
    print("Training done.\n")
    return model


# Export class prototype hypervectors
def export_prototypes(model: models.Centroid,
                      out_file: str = "mit_bih_class_prototypes.pt"):
    """
    model.weight  shape (num_classes, dimensions) holds the class prototypes.
    We save only this tensor for lightweight inference.
    """
    class_hypervectors = model.weight.data
    torch.save(class_hypervectors, out_file)
    print(f"Exported class prototypes  →  {out_file}")
    print(f"Shape: {class_hypervectors.shape}\n")
    return class_hypervectors

#
# Distance / similarity functions
#
def predict_cosine(test_samples: torch.Tensor,
                   prototypes: torch.Tensor) -> torch.Tensor:
    """Cosine similarity"""
    sims = torch.nn.functional.cosine_similarity(
        test_samples.unsqueeze(1),
        prototypes.unsqueeze(0), 
        dim=2
    )
    # Find the class with the highest cosine similarity for each sample
    return sims.argmax(dim=1)


def predict_dot(test_samples: torch.Tensor,
                prototypes: torch.Tensor) -> torch.Tensor:
    """Dot-product similarity: argmax."""
    dots = test_samples.float() @ prototypes.float().T   # (N, C)
    return dots.argmax(dim=1)


def predict_hamming(test_samples: torch.Tensor,
                    prototypes: torch.Tensor) -> torch.Tensor:
    """
    Hamming distance between binarised vectors (sign encoding):
    convert to {0,1}, count mismatches, predict class with FEWEST mismatches.
    """
    bin_samples    = (test_samples.float() >= 0).float()   # (N, D)
    bin_prototypes = (prototypes.float() >= 0).float()     # (C, D)
    # pairwise Hamming:  (N,1,D) XOR (1,C,D) -> (N,C)
    diffs = (bin_samples.unsqueeze(1) - bin_prototypes.unsqueeze(0)).abs()
    hamming = diffs.sum(dim=2)                             # (N, C)
    return hamming.argmin(dim=1)


def predict_euclidean(test_samples: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Euclidean (L2) distance: argmin."""
    dists = torch.cdist(test_samples.float(), prototypes.float(), p=2)  # (N, C)
    return dists.argmin(dim=1)


def predict_manhattan(test_samples: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Manhattan (L1) distance"""
    dists = torch.cdist(test_samples.float(), prototypes.float(), p=1)  # (N, C)
    return dists.argmin(dim=1)


DISTANCE_FNS = {
    "Cosine similarity":   predict_cosine,
    "Dot-product":         predict_dot,
    "Hamming distance":    predict_hamming,
    "Euclidean distance":  predict_euclidean,
    "Manhattan distance":  predict_manhattan,
}


# Evaluate a set of predictions and print metrics
def evaluate(preds: torch.Tensor, labels: torch.Tensor, method_name: str):
    y_true = labels.numpy()
    y_pred = preds.numpy()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score       (y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Method: {method_name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%  (macro)")
    print(f"  Recall    : {rec*100:.2f}%  (macro)")
    print(f"  F1 Score  : {f1*100:.2f}%  (macro)")
    print()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    return {"method": method_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PT_FILE   = "hdc_dataset.pt"
    PROTO_FILE = "mit_bih_class_prototypes.pt"

    # --- Load ---
    samples, labels, dimensions = load_hdc_dataset(PT_FILE)

    # --- Split 80 / 20 (reproducible) ---
    torch.manual_seed(42)
    dataset    = torch.utils.data.TensorDataset(samples, labels)
    n_train    = int(0.8 * len(dataset))
    n_test     = len(dataset) - n_train
    train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])

    # Extract tensors from the split
    train_samples = samples[train_ds.indices]
    train_labels  = labels [train_ds.indices]
    test_samples  = samples[test_ds.indices]
    test_labels   = labels [test_ds.indices]

    print(f"Train: {len(train_samples)}  |  Test: {len(test_samples)}\n")

    # --- Train Centroid model ---
    model = train_centroid(train_samples, train_labels,
                           dimensions=dimensions, num_classes=5)

    # --- Export prototypes ---
    prototypes = export_prototypes(model, out_file=PROTO_FILE)

    # --- Reload prototypes (simulates pure-inference scenario) ---
    print(f"Reloading prototypes from {PROTO_FILE}...")
    prototypes = torch.load(PROTO_FILE, weights_only=True)
    print(f"  Shape: {prototypes.shape}\n")

    # --- Benchmark every distance metric ---
    summary = []
    with torch.no_grad():
        for name, fn in DISTANCE_FNS.items():
            preds = fn(test_samples, prototypes)
            metrics = evaluate(preds, test_labels, name)
            summary.append(metrics)

    # --- Summary table ---
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    header = f"{'Method':<25} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}"
    print(header)
    print("-" * 60)
    for m in summary:
        print(f"{m['method']:<25} "
              f"{m['accuracy']*100:>6.2f}% "
              f"{m['precision']*100:>6.2f}% "
              f"{m['recall']*100:>6.2f}% "
              f"{m['f1']*100:>6.2f}%")
    print("="*60)
