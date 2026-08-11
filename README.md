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
python main.py
```

Because standard HDC thrives on orthogonality, continuous sensor data requires a specific architectural approach. We encode the 360 sample ECG windows into static high dimensional vectors.

To generate the encoded dataset (`.pt` file):

```bash
python hdc_db.py
```

You can easily modify the script to change the dimensionality (default: `10,000`) or the number of quantization levels (default: `100`).

## References & Sources

- **HDC Encoding Theory:** [Encoding Continuous Variables in Hyperdimensional Computing](https://arxiv.org/html/2411.07252v1)
- **ECG Classification with HDC:** [Hyperdimensional Computing for ECG Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7832330/)
- **Nature Article:** [Nature 2018](https://www.nature.com/articles/s41591-018-0268-3)

### Citation

> Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z. Available from: https://rdcu.be/faatM