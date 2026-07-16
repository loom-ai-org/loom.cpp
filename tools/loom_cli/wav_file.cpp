#include "wav_file.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace loom_cli {

namespace {

uint32_t read_u32le(std::istream& f) {
    unsigned char b[4] = {0, 0, 0, 0};
    f.read(reinterpret_cast<char*>(b), 4);
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
           (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}

uint16_t read_u16le(std::istream& f) {
    unsigned char b[2] = {0, 0};
    f.read(reinterpret_cast<char*>(b), 2);
    return static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
}

std::string read_tag(std::istream& f) {
    char b[4];
    f.read(b, 4);
    return std::string(b, 4);
}

} // namespace

std::vector<float> load_wav_pcm16_mono_16k(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("load_wav: cannot open '" + path + "'");
    }
    if (read_tag(f) != "RIFF") {
        throw std::runtime_error("load_wav: '" + path + "' is not a RIFF file");
    }
    read_u32le(f); // overall RIFF chunk size, unused
    if (read_tag(f) != "WAVE") {
        throw std::runtime_error("load_wav: '" + path + "' is not a WAVE file");
    }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    bool have_fmt = false;
    std::vector<int16_t> pcm;

    while (f) {
        const std::string tag = read_tag(f);
        if (!f) break;
        const uint32_t chunk_size = read_u32le(f);

        if (tag == "fmt ") {
            audio_format = read_u16le(f);
            num_channels = read_u16le(f);
            sample_rate = read_u32le(f);
            read_u32le(f); // byte rate, unused
            read_u16le(f); // block align, unused
            bits_per_sample = read_u16le(f);
            constexpr uint32_t kConsumed = 16; // bytes read for the fields above
            if (chunk_size > kConsumed) f.seekg(chunk_size - kConsumed, std::ios::cur);
            have_fmt = true;
        } else if (tag == "data") {
            if (!have_fmt) {
                throw std::runtime_error("load_wav: '" + path + "' has a 'data' chunk before 'fmt '");
            }
            if (audio_format != 1 || bits_per_sample != 16) {
                throw std::runtime_error("load_wav: '" + path + "' is not 16-bit PCM");
            }
            pcm.resize(chunk_size / sizeof(int16_t));
            f.read(reinterpret_cast<char*>(pcm.data()), chunk_size);
            break; // ignore any chunks after 'data' (e.g. trailing metadata)
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
        if (chunk_size % 2 == 1) f.seekg(1, std::ios::cur); // RIFF chunks are word-aligned
    }

    if (pcm.empty()) {
        throw std::runtime_error("load_wav: '" + path + "' has no 'data' chunk");
    }
    if (sample_rate != 16000) {
        throw std::runtime_error("load_wav: '" + path + "' is " + std::to_string(sample_rate) +
                                  "Hz; this model requires 16000Hz (no resampling implemented)");
    }

    const uint16_t channels = std::max<uint16_t>(num_channels, 1);
    const size_t n_frames = pcm.size() / channels;
    std::vector<float> out(n_frames);
    for (size_t i = 0; i < n_frames; ++i) {
        out[i] = static_cast<float>(pcm[i * channels]) / 32768.0f; // first (left) channel only
    }
    return out;
}

} // namespace loom_cli
