# Heartbeat Classification Using Hyperdimensional Computing for ECG Signals
Daris Idirene

This repository provides a complete pipeline for processing the **MIT-BIH Arrhythmia Database** and encoding it into Hyperdimensional Vectors (Hypervectors) for use with Hyperdimensional Computing (HDC) models via the [`torchhd`](https://github.com/hyperdimensional-computing/torchhd) library.


## Dataset: MIT-BIH Arrhythmia Database

The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979. Each recording is 23.5 hours long and has a sampling rate of 360 Hz. 

For more information, visit the [PhysioNet Dataset Page](https://physionet.org/content/mitdb/1.0.0/).

Download using the following commands:

```bash
wget -nc -r -np https://physionet.org/files/mitdb/1.0.0/
```

## Setup & Installation

```bash
pip install -r requirements.txt
```


## Usage

The `main.py` script loads the raw WFDB dataset, extracts the heartbeats, maps the labels, and builds a standard PyTorch `MITBIHDataset`.

```bash
python create_db.py
```

Because standard HDC thrives on orthogonality, continuous sensor data requires a specific architectural approach. We encode the 360 sample ECG windows into static high dimensional vectors.

To generate the encoded dataset (`.pt` file):

```bash
python hdc_db.py
```

You can easily modify the script to change the dimensionality (default: `10,000`) or the number of quantization levels (default: `100`).

### 3. HDC Classification & Benchmark (`class_hdc_impl.py`)

In standard Hyperdimensional Computing, a classifier learns by bundling all the training hypervectors of a given class into a single "prototype" vector. Once trained, these class prototypes can be extracted, saved, and used for lightweight inference on new data with some similarity measurements based on these class prototypes.

To train the centroid prototypes, export them, and benchmark their performance across different distance metrics, run:

```bash
python class_hdc_impl.py
```

**Benchmark Results (1,000 Dimensions):**

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| **Euclidean distance** | 82.55% | 18.12% |
| **Dot-product** | 82.39% | 18.43% |
| Hamming distance | 7.70% | 3.28% |
| Cosine similarity | 3.04% | 1.58% |
| Manhattan distance | 2.56% | 1.00% |

## References & Sources

- **HDC Encoding Theory:** [Encoding Continuous Variables in Hyperdimensional Computing](https://arxiv.org/html/2411.07252v1)
- **ECG Classification with HDC:** [Hyperdimensional Computing for ECG Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7832330/)
- **Nature Article:** [Nature 2018](https://www.nature.com/articles/s41591-018-0268-3)

### Citation

> Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z. Available from: https://rdcu.be/faatM