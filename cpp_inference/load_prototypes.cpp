// cpp_inference/load_prototypes.cpp
//
// Host-side validation that the .pt files load in C++ with LibTorch and that
// the bipolar quantization + Hamming inference match the Python flow.
//
// IMPORTANT: this runs on your development machine. The bare-metal RISC-V
// target cannot run LibTorch — for the accelerator flow, use
// export_prototypes_c.py, which emits a C header compiled straight into the
// test binary. Use this program to sanity-check the model in C++ first.
//
// Build (after downloading LibTorch from https://pytorch.org):
//   mkdir build && cd build
//   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
//   make && ./load_prototypes

#include <torch/torch.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static std::vector<char> readBytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// torch.sign, with exact zeros (majority ties) resolved to +1 — identical to
// bipolarize() in export_prototypes_c.py.
static torch::Tensor bipolarize(const torch::Tensor& t) {
    auto s = torch::sign(t.to(torch::kFloat));
    return torch::where(s == 0, torch::ones_like(s), s);
}

int main() {
    // Files written by torch.save() are pickle archives, NOT TorchScript —
    // torch::jit::load() will reject them. Use torch::pickle_load() instead.
    auto dict    = torch::pickle_load(readBytes("../hdc_dataset.pt")).toGenericDict();
    auto samples = dict.at("samples").toTensor();   // (N, D) float
    auto labels  = dict.at("labels").toTensor();    // (N,)   long

    // Class prototypes exported by class_hdc_impl.py are a plain tensor .pt.
    // NOTE: for the bipolar flow you normally retrain in the binary domain
    // (export_prototypes_c.py) instead of loading the real-valued file —
    // loaded here to demonstrate the API.
    auto protos = torch::pickle_load(readBytes("../mit_bih_class_prototypes.pt")).toTensor();

    std::cout << "samples " << samples.sizes() << "  protos " << protos.sizes() << "\n";

    auto binSamples = bipolarize(samples);
    auto binProtos  = bipolarize(protos);

    // Hamming distance = number of mismatched dimensions; argmin over classes.
    auto mism  = (binSamples.unsqueeze(1) != binProtos.unsqueeze(0)).sum(2);  // (N, C)
    auto preds = mism.argmin(1);                                              // (N,)
    auto acc   = (preds == labels).to(torch::kFloat).mean().item<float>();

    std::cout << "Bipolar Hamming accuracy: " << acc * 100.0f << "%\n";
    std::cout << "(Compare against export_prototypes_c.py output before going to RTL.)\n";
    return 0;
}
