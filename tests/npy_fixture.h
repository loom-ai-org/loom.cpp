#pragma once

// Reading and writing the little `.npy` float32 fixtures the reference tests compare against.
//
// Most tests here already carry a private `read_npy_f32` copy for the numpy references their
// `reference_forward_*.py` generators produce; this header exists for the fixtures that have no Python
// generator -- the frozen waveforms of the retired per-model C++ drivers (P4.0.8, E.3), where the
// producing code is deleted in the same commit that starts consuming the file. `.npy` is used for those
// too rather than a raw dump, so `numpy.load` opens them and the shape travels with the data.
//
// Deliberately minimal: 1-D float32, little-endian, C-order, version 1.0 -- which is all any fixture
// here needs, and all `write_npy_f32` emits.

#include "test_util.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace loom_test {

// Reads a 1-D (or flattened) float32 .npy. `shape_out` receives the declared dimensions.
inline std::vector<float> read_npy_f32(const std::string& path, std::vector<int64_t>& shape_out) {
    std::ifstream f(path, std::ios::binary);
    LOOM_CHECK(static_cast<bool>(f));
    char magic[6];
    f.read(magic, 6);
    f.ignore(2); // version
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    const size_t shape_pos = header.find("'shape':");
    const size_t paren_open = header.find('(', shape_pos);
    const size_t paren_close = header.find(')', paren_open);
    std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);
    shape_out.clear();
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::string trimmed;
        for (char c : tok) if (c != ' ') trimmed += c;
        if (!trimmed.empty()) shape_out.push_back(std::stoll(trimmed));
    }
    int64_t total = 1;
    for (int64_t d : shape_out) total *= d;
    std::vector<float> data(static_cast<size_t>(total));
    f.read(reinterpret_cast<char*>(data.data()), total * static_cast<int64_t>(sizeof(float)));
    return data;
}

// Writes `data` as a 1-D float32 .npy. Returns false if the file could not be opened.
inline bool write_npy_f32(const std::string& path, const std::vector<float>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream dict;
    dict << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << data.size() << ",), }";
    std::string header = dict.str();
    // The header (magic + version + length field + dict) must be a multiple of 64 bytes, padded with
    // spaces and terminated by '\n' -- that is the format's own alignment rule, not a preference.
    size_t unpadded = 6 + 2 + 2 + header.size() + 1;
    size_t pad = (64 - (unpadded % 64)) % 64;
    header.append(pad, ' ');
    header.push_back('\n');
    const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    const char version[2] = {1, 0};
    const auto header_len = static_cast<uint16_t>(header.size());
    f.write(magic, 6);
    f.write(version, 2);
    f.write(reinterpret_cast<const char*>(&header_len), 2);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    return static_cast<bool>(f);
}

} // namespace loom_test
