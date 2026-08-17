# C++ inference with LibTorch (host-side validation)

This folder shows how to load the HDC model in C++ using LibTorch. It is for
**validating the quantization on your development machine** — the bare-metal
RISC-V target of `hyperdim-rocc` cannot run LibTorch, so the accelerator flow
uses `export_prototypes_c.py` (repo root), which packs the bipolar prototypes
into a C header instead.

## Key API note

Files written by `torch.save()` (like `hdc_dataset.pt` and
`mit_bih_class_prototypes.pt`) are **pickle archives, not TorchScript**.
`torch::jit::load()` will reject them. Use `torch::pickle_load()`:

```cpp
#include <torch/csrc/jit/serialization/pickle.h>

auto dict    = torch::pickle_load(bytes).toGenericDict();   // for the dataset dict
auto samples = dict.at("samples").toTensor();
auto protos  = torch::pickle_load(bytes).toTensor();        // for a plain tensor
```

## Build & run

1. Download LibTorch (CPU build is fine) from https://pytorch.org and unzip.
2. Build:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
./load_prototypes
```

3. Compare the printed bipolar Hamming accuracy against the one reported by
   `export_prototypes_c.py`. If they match, the C++ and Python quantization
   flows agree and the header can be trusted for hardware tests.
